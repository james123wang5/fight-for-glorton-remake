const CACHE_NAME = "glorton-hosted-3e302791df93";
const SHELL = ["./", "./index.html", "./game.html", "./manifest.webmanifest", "./apple-touch-icon.png", "./favicon.png"];
const CHUNKS = ["chunks/app-000.bin","chunks/app-001.bin","chunks/app-002.bin","chunks/app-003.bin","chunks/app-004.bin"];
const APK_SIZE = 37359606;
const PYGAME_CDN = "pygame-web.github.io";

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys().then((keys) => Promise.all(
    keys.filter((key) => key.startsWith("glorton-hosted-") && key !== CACHE_NAME).map((key) => caches.delete(key))
  )));
  self.clients.claim();
});

async function chunkResponse(name) {
  const cache = await caches.open(CACHE_NAME);
  const request = new Request(new URL(name, self.registration.scope), {credentials: "same-origin"});
  let response = await cache.match(request);
  if (!response) {
    response = await fetch(request);
    if (!response.ok) throw new Error(`Chunk ${name} returned ${response.status}`);
    await cache.put(request, response.clone());
  }
  return response;
}

function streamedApk() {
  const body = new ReadableStream({
    async start(controller) {
      try {
        for (const name of CHUNKS) {
          const response = await chunkResponse(name);
          if (!response.body) throw new Error(`Chunk ${name} has no body`);
          const reader = response.body.getReader();
          while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            controller.enqueue(value);
          }
        }
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    }
  });
  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Length": String(APK_SIZE),
      "Cache-Control": "no-store"
    }
  });
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin === self.location.origin && url.pathname.endsWith("/game/app.apk")) {
    event.respondWith(streamedApk());
    return;
  }
  if (request.headers.has("range")) return;
  const cacheable = url.origin === self.location.origin || url.hostname === PYGAME_CDN;
  if (!cacheable) return;
  event.respondWith(caches.match(request).then((cached) => cached || fetch(request).then((response) => {
    if (response.ok || response.type === "opaque") {
      const copy = response.clone();
      caches.open(CACHE_NAME).then((cache) => cache.put(request, copy)).catch(() => {});
    }
    return response;
  }).catch(() => caches.match("./index.html"))));
});
