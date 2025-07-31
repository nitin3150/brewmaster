# import asyncio
# import websockets
# import json
# from model import BrewMastersModel

# # Create a single, persistent instance of your game model
# game_model = BrewMastersModel()

# async def handler(websocket, path):
#     """Handles communication with a single client."""
#     print("Front-end client connected.")
#     try:
#         # Send the initial game state on connection

#         await websocket.send(game_model.get_state_as_json())
#         async for message in websocket:
#             human_decisions = json.loads(message)
#             print(f"Received human decisions: {human_decisions}")
#             # Process one full turn for both teams
#             game_model.step(human_decisions)
#             # Send the new, updated game state back to the client
#             await websocket.send(game_model.get_state_as_json())
#     except websockets.exceptions.ConnectionClosed:
#         print("Client disconnected.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         import traceback
#         traceback.print_exc()
# async def main():
#     """Starts the WebSocket server."""
#     async with websockets.serve(handler, "0.0.0.0", 8766):
#         print("BrewMasters MAS Server started on port 8766...")
#         await asyncio.Future()  # Run forever
# if __name__ == "__main__":
#     asyncio.run(main())
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from model import BrewMastersModel
import json

app = FastAPI()

# Enable CORS (so frontend like Hoppscotch, localhost:3000, etc. can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a persistent model instance
game_model = BrewMastersModel()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected via WebSocket")

    try:
        # Send initial game state
        await websocket.send_text(game_model.get_state_as_json())

        while True:
            message = await websocket.receive_text()
            human_decisions = json.loads(message)
            print(f"Received human decisions: {human_decisions}")

            # Run a simulation turn
            game_model.step(human_decisions)

            # Send updated state back
            await websocket.send_text(game_model.get_state_as_json())

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()