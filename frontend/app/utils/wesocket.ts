import { w3cwebsocket as W3CWebSocket } from "websocket";

let client: W3CWebSocket | null = null;

export const getWebSocketClient = () => {
  if (typeof window !== 'undefined' && (!client || client.readyState === W3CWebSocket.CLOSED)) {
    console.log("Creating new WebSocket client...");
    client = new W3CWebSocket("ws://127.0.0.1:8000/ws");

    client.onopen = () => {
      console.log("WebSocket Client Connected");
    };

    client.onerror = (error: Error) => {
      console.error("WebSocket Error:", error);
    };

    client.onclose = () => {
      console.log("WebSocket Client Closed");
    };
  }
  return client;
};
