'use client';
import { useEffect, useState } from "react";
import Controls from "./controls";
import { getWebSocketClient } from "../utils/wesocket";
import { IMessageEvent, w3cwebsocket as W3CWebSocket } from "websocket";
import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
 
interface TeamPanelProps {
    title: string;
    totalProfit: number;
    profitThisTurn: number;
    inventory: number;
    projectedDemand: number;
    children?: React.ReactNode;
    bgColor: string;
    textColor: string;
}

const TeamPanel: React.FC<TeamPanelProps> = ({ title, totalProfit, profitThisTurn, inventory, projectedDemand, children, bgColor, textColor }) => (
    <div className={`${bgColor} w-1/2 p-6 flex flex-col`}>
        <h1 className={`text-2xl font-bold ${textColor} mb-4`}>{title}</h1>
        <div className="flex-grow">
            <p className="text-4xl font-bold">${totalProfit}</p>
            <p className="text-lg text-gray-600">Total Profit</p>

            <p className="text-2xl font-bold mt-4">${profitThisTurn}</p>
            <p className="text-md text-gray-600">Profit This Turn</p>

            <p className="text-2xl font-bold mt-4">{inventory} units</p>
            <p className="text-md text-gray-600">Current Inventory</p>

            <p className="text-2xl font-bold mt-4">{projectedDemand} units</p>
            <p className="text-md text-gray-600">Projected Demand</p>
        </div>
        {children}
    </div>
);

interface GraphData {
    name: string;
    green: number;
    blue: number;
}

interface OrderVolumeGraphProps {
    data: GraphData[];
}

const OrderVolumeGraph: React.FC<OrderVolumeGraphProps> = ({ data }) => (
    <div className="w-full h-64 bg-gray-100 p-4 mt-4">
        <h2 className="text-lg font-bold mb-2">Order Volume Graph</h2>
        <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="green" stroke="#82ca9d" activeDot={{ r: 8 }} />
                <Line type="monotone" dataKey="blue" stroke="#8884d8" />
            </LineChart>
        </ResponsiveContainer>
    </div>
);

interface EventLogProps {
    logs: string[];
}

const EventLog: React.FC<EventLogProps> = ({ logs }) => (
    <div className="w-full h-48 bg-gray-800 text-white font-mono p-4 mt-4 overflow-y-auto">
        <h2 className="text-lg font-bold mb-2 text-gray-300">Event Log</h2>
        {logs.map((log, index) => (
            <p key={index} className="text-sm">{`> ${log}`}</p>
        ))}
    </div>
);


function Dashboard() {
    const [logs, setLogs] = useState<string[]>([]);
    const [client, setClient] = useState<W3CWebSocket | null>(null);

    // Placeholder data
    const [greenTeamData, setGreenTeamData] = useState({ totalProfit: 12000, profitThisTurn: 500, inventory: 150, projectedDemand: 200 });
    const [blueTeamData, setBlueTeamData] = useState({ totalProfit: 15000, profitThisTurn: 750, inventory: 200, projectedDemand: 250 });
    const [graphData, setGraphData] = useState([
        { name: 'Turn 1', green: 100, blue: 120 },
        { name: 'Turn 2', green: 110, blue: 130 },
        { name: 'Turn 3', green: 105, blue: 125 },
        { name: 'Turn 4', green: 120, blue: 140 },
    ]);


    useEffect(() => {
        const wsClient = getWebSocketClient();
        setClient(wsClient);

        if (wsClient) {
            wsClient.onmessage = (message: IMessageEvent) => {
                if (typeof message.data === 'string') {
                    console.log("Received message from server: ", message.data);
                    setLogs(prevLogs => [...prevLogs, message.data as string]);
                    // Here you would parse the message and update team data
                    // e.g. const data = JSON.parse(message.data);
                    // setGreenTeamData(...);
                    // setBlueTeamData(...);
                }
            };
        }

        return () => {
            if (wsClient) {
                wsClient.onmessage = () => {};
            }
        };
    }, []);
    
    return(
        <div className="dashboard flex flex-col h-screen bg-gray-50">
            <div className="flex flex-grow">
                <TeamPanel 
                    title="🟩 Green Team (Human Player)"
                    totalProfit={greenTeamData.totalProfit}
                    profitThisTurn={greenTeamData.profitThisTurn}
                    inventory={greenTeamData.inventory}
                    projectedDemand={greenTeamData.projectedDemand}
                    bgColor="bg-green-50"
                    textColor="text-green-700"
                >
                    <Controls client={client} />
                </TeamPanel>

                <div className="border-r-2 border-gray-300"></div>

                <TeamPanel
                    title="🟦 Blue Team (MAS)"
                    totalProfit={blueTeamData.totalProfit}
                    profitThisTurn={blueTeamData.profitThisTurn}
                    inventory={blueTeamData.inventory}
                    projectedDemand={blueTeamData.projectedDemand}
                    bgColor="bg-blue-50"
                    textColor="text-blue-700"
                />
            </div>
            <div className="flex-shrink-0 p-6">
                <OrderVolumeGraph data={graphData} />
                <EventLog logs={logs} />
            </div>
        </div>
    );
}

export default Dashboard;
