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

# Store game instances per connection
game_instances = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected via WebSocket")
    
    # Create a new game instance for this connection
    connection_id = id(websocket)
    game_instances[connection_id] = BrewMastersModel()
    game_model = game_instances[connection_id]

    try:
        # Send initial game state
        await websocket.send_text(game_model.get_state_as_json())

        while True:
            message = await websocket.receive_text()
            
            # Check if it's a restart command
            if message == '{"restart": true}':
                # Create a fresh game instance
                game_instances[connection_id] = BrewMastersModel()
                game_model = game_instances[connection_id]
                await websocket.send_text(game_model.get_state_as_json())
                continue
            
            # Otherwise, process as normal turn
            human_decisions = json.loads(message)
            print(f"Received human decisions: {human_decisions}")

            # Run a simulation turn
            game_model.step(human_decisions)

            # Send updated state back
            await websocket.send_text(game_model.get_state_as_json())

    except WebSocketDisconnect:
        print("Client disconnected")
        # Clean up the game instance
        if connection_id in game_instances:
            del game_instances[connection_id]
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        if connection_id in game_instances:
            del game_instances[connection_id]