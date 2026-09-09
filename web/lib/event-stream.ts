/** Read NDJSON across arbitrary network chunks, always releasing the stream. */
export async function readEventStream<T extends { type: string }>(
  response: Response,
  onEvent: (event: T) => void,
) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: unknown } | null;
    throw new Error(
      typeof body?.detail === 'string'
        ? body.detail
        : `Request failed (${response.status})`,
    );
  }
  if (!response.body) throw new Error('The response stream is unavailable.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const processLine = (line: string) => {
    if (!line.trim()) return;
    const value = JSON.parse(line);
    if (!value || typeof value.type !== 'string')
      throw new Error('The server returned an invalid event.');
    onEvent(value as T);
  };
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      lines.forEach(processLine);
      if (done) {
        processLine(buffer);
        break;
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
