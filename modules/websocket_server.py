import asyncio
import websockets
import json
import threading
from typing import Optional

class GazeWebSocketServer:
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients = set()
        self.server = None
        self.loop = None
        self.thread = None

    async def register(self, websocket):
        """Register a new client."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")

    async def unregister(self, websocket):
        """Unregister a client."""
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")

    async def handler(self, websocket, path=None):
        """Handle client connections."""
        await self.register(websocket)
        try:
            async for message in websocket:
                # Echo received messages or handle commands
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def broadcast(self, message):
        """Broadcast message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )

    def broadcast_gaze_data(self, gaze_data):
        """Thread-safe method to broadcast gaze data."""
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.broadcast(json.dumps(gaze_data)),
                self.loop
            )

    def start_server(self):
        """Start the WebSocket server in a separate thread."""
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def start():
                self.server = await websockets.serve(
                    self.handler,
                    "localhost",
                    self.port
                )
                print(f"WebSocket server started on ws://localhost:{self.port}")
                await self.server.wait_closed()

            self.loop.run_until_complete(start())

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

    def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
        if self.loop:
            self.loop.stop()