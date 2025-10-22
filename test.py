import asyncio

import websockets


async def main():
    uri = "ws://localhost:5001"
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        try:
            async for message in websocket:
                print("Peak strength:", message)
        except websockets.ConnectionClosed:
            print("Connection closed")


if __name__ == "__main__":
    asyncio.run(main())
