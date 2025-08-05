import { w3cwebsocket as W3CWebSocket } from "websocket";

export const getWebSocketClient = (): W3CWebSocket | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  
  console.log("Creating new WebSocket client...");
  const client = new W3CWebSocket("ws://127.0.0.1:8000/ws");

  client.onopen = () => {
    console.log("WebSocket Client Connected");
  };

  client.onerror = (error: Error) => {
    console.error("WebSocket Error:", error);
  };

  client.onclose = () => {
    console.log("WebSocket Client Closed");
  };

  return client;
};
