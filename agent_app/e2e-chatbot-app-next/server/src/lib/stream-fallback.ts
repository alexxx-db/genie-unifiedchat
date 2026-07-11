import {
  streamText,
  type LanguageModelUsage,
  type UIMessageStreamWriter,
} from 'ai';

/**
 * Reads all chunks from a UI message stream, forwarding non-error parts to the
 * writer. Returns whether the stream encountered any errors.
 */
export async function drainStreamToWriter(
  uiStream: ReadableStream,
  writer: UIMessageStreamWriter,
): Promise<{ failed: boolean; errorText?: string }> {
  const reader = uiStream.getReader();
  let receivedTextChunk = false;

  try {
    for (
      let chunk = await reader.read();
      !chunk.done;
      chunk = await reader.read()
    ) {
      if (chunk.value.type === 'error') {
        if (!receivedTextChunk) {
          console.error(
            'Error before first text chunk, triggering fallback:',
            chunk.value.errorText,
          );
          return { failed: true, errorText: chunk.value.errorText };
        }
        console.error(
          'Mid-stream error, forwarding to client:',
          chunk.value.errorText,
        );
        writer.write(chunk.value);
      } else {
        if (!receivedTextChunk && chunk.value.type.startsWith('text-')) {
          receivedTextChunk = true;
        }
        writer.write(chunk.value);
      }
    }
  } catch (readError) {
    if (!receivedTextChunk) {
      console.error('Stream read error before first text chunk:', readError);
      return { failed: true };
    }
    console.error('Mid-stream read error:', readError);
  } finally {
    reader.releaseLock();
  }

  return { failed: false };
}

/**
 * Retries with streamText so the fallback preserves incremental UI streaming
 * instead of waiting for a full generateText result before emitting chunks.
 */
export async function fallbackToStreamText(
  params: Parameters<typeof streamText>[0],
  writer: UIMessageStreamWriter,
): Promise<{
  usage?: LanguageModelUsage;
  traceId?: string;
  clarificationData?: { reason: string; options: string[] };
} | undefined> {
  try {
    let traceId: string | undefined;
    let clarificationData:
      | {
          reason: string;
          options: string[];
        }
      | undefined;
    let usage: LanguageModelUsage | undefined;

    const fallback = streamText({
      ...params,
      includeRawChunks: true,
      onChunk: ({ chunk }) => {
        params.onChunk?.({ chunk });

        if (chunk.type !== 'raw') {
          return;
        }

        const raw = chunk.rawValue as any;
        if (raw?.type === 'response.output_item.done') {
          const traceIdFromChunk = raw?.databricks_output?.trace?.info?.trace_id;
          if (typeof traceIdFromChunk === 'string') {
            traceId = traceIdFromChunk;
          }
        }

        if (!traceId && typeof raw?.trace_id === 'string') {
          traceId = raw.trace_id;
        }

        if (raw?.databricks_output?.clarification) {
          clarificationData = raw.databricks_output.clarification;
        }
      },
      onFinish: (event) => {
        usage = event.totalUsage;
        // Forward the full finish event; params.onFinish requires the complete
        // StepResult payload, not just { usage }.
        params.onFinish?.(event);
      },
    });

    const fallbackUiStream = fallback.toUIMessageStream({
      sendReasoning: true,
      sendSources: true,
      sendFinish: false,
      onError: (error) => {
        const msg = error instanceof Error ? error.message : String(error);
        writer.onError?.(error);
        return msg;
      },
    });

    const { failed } = await drainStreamToWriter(fallbackUiStream, writer);
    if (failed) {
      throw new Error('streamText fallback failed before first text chunk');
    }

    return { usage, traceId, clarificationData };
  } catch (fallbackError) {
    console.error(
      '[fallbackToStreamText] streamText fallback also failed:',
      fallbackError,
    );
    const errorMessage =
      fallbackError instanceof Error
        ? fallbackError.message
        : String(fallbackError);
    writer.write({ type: 'data-error', data: errorMessage });
    return undefined;
  }
}
