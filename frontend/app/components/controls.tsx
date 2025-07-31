'use client';
import { useState } from 'react';
import { w3cwebsocket as W3CWebSocket } from 'websocket';

interface ControlsProps {
    client: W3CWebSocket | null;
}

function Controls({ client }: ControlsProps) {
    const [price, setPrice] = useState<number>(10);
    const [production, setProduction] = useState<number>(100);
    const [marketing, setMarketing] = useState<number>(500);

    const handleEndTurn = () => {
        if (client && client.readyState === client.OPEN) {
            console.log("End turn clicked");
            client.send(JSON.stringify({ price, production, marketing }));
        } else {
            console.error("WebSocket is not connected.");
        }
    };

    return (
        <div className="control-panel bg-white rounded-lg shadow-md p-4 space-y-4">
            <h2 className="text-lg font-bold text-gray-800">Control Panel</h2>

            <div>
                <label className="block text-sm font-medium text-gray-700">
                    Price: ${price}
                </label>
                <input
                    type="range"
                    min="8"
                    max="15"
                    value={price}
                    onChange={(e) => setPrice(Number(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
            </div>

            <div>
                <label className="block text-sm font-medium text-gray-700">
                    Production Target (units)
                </label>
                <input
                    type="number"
                    value={production}
                    onChange={(e) => setProduction(Number(e.target.value))}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                    placeholder="Enter units"
                />
            </div>

            <div>
                <label className="block text-sm font-medium text-gray-700">
                    Marketing Spend
                </label>
                <div className="flex justify-around p-2 bg-gray-100 rounded-lg">
                    {[0, 500, 2000].map((amount) => (
                        <button
                            key={amount}
                            onClick={() => setMarketing(amount)}
                            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                                marketing === amount
                                    ? "bg-green-600 text-white"
                                    : "bg-white text-gray-700 hover:bg-gray-200"
                            }`}
                        >
                            ${amount}
                        </button>
                    ))}
                </div>
            </div>

            <button
                onClick={handleEndTurn}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 ease-in-out transform hover:scale-105"
            >
                End Turn
            </button>
        </div>
    );
}

export default Controls;
