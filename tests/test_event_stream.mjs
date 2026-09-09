import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readEventStream } from '../web/lib/event-stream.ts';

test('reads UTF-8 events split across chunks and a final line without newline', async () => {
  const input = [{ type: 'status', message: 'Fold → restart' }, { type: 'complete' }];
  const bytes = new TextEncoder().encode(input.map(JSON.stringify).join('\n'));
  const body = new ReadableStream({
    start(controller) {
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      controller.close();
    },
  });
  const actual = [];
  await readEventStream(new Response(body), (event) => actual.push(event));
  assert.deepEqual(actual, input);
  assert.equal(body.locked, false);
});

test('cancels and releases the stream when the event handler fails', async () => {
  let cancelled = false;
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('{"type":"error"}\n'));
    },
    cancel() { cancelled = true; },
  });
  await assert.rejects(readEventStream(new Response(body), () => {
    throw new Error('Stop processing');
  }), /Stop processing/);
  assert.equal(cancelled, true);
  assert.equal(body.locked, false);
});

test('rejects malformed events and preserves HTTP error details', async () => {
  await assert.rejects(readEventStream(new Response('{"unrecognized":true}'), () => {}), /invalid event/);
  await assert.rejects(readEventStream(new Response('{"detail":"Already running"}', { status: 409 }), () => {}), /Already running/);
  await assert.rejects(readEventStream(new Response(null), () => {}), /stream is unavailable/);
});

test('propagates a disconnected stream and releases its reader', async () => {
  const body = new ReadableStream({ start(controller) { controller.error(new Error('Disconnected')); } });
  await assert.rejects(readEventStream(new Response(body), () => {}), /Disconnected/);
  assert.equal(body.locked, false);
});
