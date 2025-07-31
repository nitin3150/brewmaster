'use client';
import { useEffect, useState } from "react";
import { getWebSocketClient } from "../utils/wesocket";
import { IMessageEvent, w3cwebsocket as W3CWebSocket } from "websocket";
import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import Controls from "./controls";

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
    <div className={`${bgColor} w-full p-6 flex flex-col`}>
        <h1 className={`text-2xl font-bold ${textColor} mb-4`}>{title}</h1>
        <div className="flex-grow">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
                <p className="text-4xl font-bold">${totalProfit.toLocaleString()}</p>
                <p className="text-lg text-gray-600">Total Profit</p>
            </motion.div>

            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.2 }}>
                <p className="text-2xl font-bold mt-4">${profitThisTurn.toLocaleString()}</p>
                <p className="text-md text-gray-600">Profit This Turn</p>
            </motion.div>

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

interface ProductionGraphProps {
    data: GraphData[];
}

const ProductionGraph: React.FC<ProductionGraphProps> = ({ data }) => (
    <div className="w-full h-64 bg-gray-100 p-4 mt-4">
        <h2 className="text-lg font-bold mb-2">Production Target Over Time</h2>
        <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="green" stroke="#82ca9d" name="Green Team" />
                <Line type="monotone" dataKey="blue" stroke="#8884d8" name="Blue Team" />
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

    const [greenTeamData, setGreenTeamData] = useState({ totalProfit: 0, profitThisTurn: 0, inventory: 0, projectedDemand: 0, productionTarget: 0 });
    const [blueTeamData, setBlueTeamData] = useState({ totalProfit: 0, profitThisTurn: 0, inventory: 0, projectedDemand: 0, productionTarget: 0 });
    const [graphData, setGraphData] = useState<GraphData[]>([]);


    useEffect(() => {
        const wsClient = getWebSocketClient();
        setClient(wsClient);

        if (wsClient) {
            wsClient.onmessage = (message: IMessageEvent) => {
                if (typeof message.data === 'string') {
                    console.log("Received message from server: ", message.data);
                    const data = JSON.parse(message.data);

                    setGreenTeamData({
                        totalProfit: data.green_team_profit,
                        profitThisTurn: data.green_team_profit_this_turn || 0,
                        inventory: data.green_team_inventory,
                        projectedDemand: data.green_team_projected_demand || 0,
                        productionTarget: data.green_team_production_target || 0,
                    });
                    setBlueTeamData({
                        totalProfit: data.blue_team_profit,
                        profitThisTurn: data.blue_team_profit_this_turn || 0,
                        inventory: data.blue_team_inventory,
                        projectedDemand: data.blue_team_projected_demand || 0,
                        productionTarget: data.blue_team_production_target || 0,
                    });

                    setGraphData(prevData => [...prevData, {
                        name: `Turn ${data.turn}`,
                        green: data.green_team_production_target || 0,
                        blue: data.blue_team_production_target || 0,
                    }]);

                    setLogs(prevLogs => {
                        const newLogs = Array.isArray(data.event_log) ? data.event_log : [data.event_log];
                        return [...prevLogs, ...newLogs].slice(-10); // Keep last 10 logs
                    });
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
        <div className="dashboard grid grid-cols-2 gap-4 h-screen bg-gray-50 p-4">
            <div className="flex flex-col">
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
            </div>

            <div className="flex flex-col">
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

            <div className="col-span-2">
                <ProductionGraph data={graphData} />
            </div>
            <div className="col-span-2">
                <EventLog logs={logs} />
            </div>
        </div>
    );
}

export default Dashboard;
