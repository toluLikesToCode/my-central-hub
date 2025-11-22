# Plex-Style Remote Streaming Notes

## Key takeaways
- Plex uses HLS with multiple renditions, small segments, and ABR to hide WAN jitter and start playback quickly.
- Clients advertise capabilities so the server can choose direct play, direct stream (remux), or transcoding; hardware acceleration minimizes latency when transcoding.
- Persistent connections, prefetching, and buffering keep playback responsive; relay fallback exists but is capped at 2 Mbps, so direct connections are preferred.

## Recommendations for a Node/TypeScript implementation
1. **Use segmented streaming (HLS/DASH)** instead of pushing full files. Serve manifests and segment endpoints rather than a single download.
2. **Offer multiple renditions** so clients can switch bitrate when bandwidth changes. Pre-encode commonly watched titles; transcode on demand for infrequent titles or when the requested rendition is missing.
3. **Offload transcoding** to worker processes or a dedicated service that can leverage GPU/Quick Sync; keep the Node server focused on I/O.
4. **Front the service with a reverse proxy** (NGINX/Caddy/Traefik) for TLS termination, caching of static segments, and load balancing across transcoder workers.
5. **Keep connections alive and tune streaming buffers** (e.g., `highWaterMark`) to balance latency vs. memory, but rely on ABR rather than buffering alone to handle slow links.
6. **Plan for NAT traversal and relays**; fall back to a relay only when direct connections fail, and enforce a bitrate cap similar to Plex’s 2 Mbps when relayed.

## Answers to the follow-up questions
- **Q1:** Prefer a hybrid approach—pre-encode popular libraries into a ladder of renditions so playback starts instantly, and transcode on demand (with caching) for less common titles or niche device profiles.
- **Q2:** Capacity planning hinges on simultaneous remote users: a single mid-range GPU can usually handle ~3–6 1080p transcodes, while direct-play sessions are cheap. Expect to add load balancing/reverse proxying if you regularly exceed that range.
- **Q3:** Use a specialized player such as `hls.js` on the web or native HLS/DASH SDKs on mobile/TV; generic `<video>` tags will not perform ABR or rendition switching without such a client-side library.
