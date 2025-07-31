'use client';
import { useState } from 'react';
import { w3cwebsocket as W3CWebSocket } from 'websocket';

interface ControlsProps {
    client: W3CWebSocket | null;
}

function Controls({ client }: ControlsProps) {
    const [price, setPrice] = useState(10);
    const [productionTarget, setProductionTarget] = useState(50);
    const [marketingSpend, setMarketingSpend] = useState(500);

    const endTurn = () => {
        if (client) {
            const decisions = {
                price,
                productionTarget,
                marketingSpend,
            };
            client.send(JSON.stringify(decisions));
        }
    };

    return (
        <div className="bg-gray-200 p-4 rounded-lg mt-4">
            <h2 className="text-xl font-bold mb-4">Control Panel</h2>
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700">Set Price: ${price}</label>
                <input
                    type="range"
                    min="8"
                    max="15"
                    value={price}
                    onChange={(e) => setPrice(Number(e.target.value))}
                    className="w-full"
                />
            </div>
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700">Production Target: {productionTarget} units</label>
                <input
                    type="range"
                    min="0"
                    max="1000"
                    value={productionTarget}
                    onChange={(e) => setProductionTarget(Number(e.target.value))}
                    className="w-full"
                />
            </div>
            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700">Marketing Spend: ${marketingSpend}</label>
                <input
                    type="range"
                    min="0"
                    max="2000"
                    step="500"
                    value={marketingSpend}
                    onChange={(e) => setMarketingSpend(Number(e.target.value))}
                    className="w-full"
                />
                 <div className="w-full flex justify-between text-xs px-2">
                    <span>$0</span>
                    <span>$500</span>
                    <span>$1000</span>
                    <span>$1500</span>
                    <span>$2000</span>
                </div>
            </div>
            <button
                onClick={endTurn}
                className="w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
            >
                End Turn
            </button>
        </div>
    );
};

export default Controls;
