'use client';
import { useEffect, useState } from "react";
import { getWebSocketClient } from "../utils/wesocket";
import { IMessageEvent, w3cwebsocket as W3CWebSocket } from "websocket";
import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
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
    accentColor: string;
    team: 'green' | 'blue';
}

const TeamPanel: React.FC<TeamPanelProps> = ({ 
    title, 
    totalProfit, 
    profitThisTurn, 
    inventory, 
    projectedDemand, 
    children, 
    bgColor, 
    textColor,
    accentColor,
    team 
}) => (
    <motion.div 
        className={`${bgColor} w-full p-2 flex flex-col rounded-lg shadow-xl backdrop-blur-sm border border-white/20 relative overflow-hidden h-full`}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
    >
        {/* Background decoration */}
        <div className={`absolute top-0 right-0 w-32 h-32 bg-gradient-to-br ${team === 'green' ? 'from-green-400/20 to-transparent' : 'from-blue-400/20 to-transparent'} rounded-full blur-3xl`} />
        
        <h1 className={`text-base font-bold ${textColor} mb-2 flex items-center gap-1`}>
            <span className="text-lg">{team === 'green' ? '🟩' : '🟦'}</span>
            {title}
        </h1>
        
        <div className={`${children ? 'flex-1' : 'flex-grow'} space-y-2 overflow-hidden`}>
            <div className="grid grid-cols-2 gap-2">
                <motion.div 
                    className={`p-2 rounded-lg bg-white/10 backdrop-blur-sm border border-white/20 shadow-lg`}
                    initial={{ opacity: 0, y: 10 }} 
                    animate={{ opacity: 1, y: 0 }} 
                    transition={{ duration: 0.5 }}
                >
                    <p className={`text-xl font-bold ${accentColor}`}>
                        ${totalProfit.toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-600 font-medium">Total Profit</p>
                </motion.div>

                <motion.div 
                    className={`p-2 rounded-lg ${profitThisTurn >= 0 ? 'bg-green-50/50' : 'bg-red-50/50'} backdrop-blur-sm border ${profitThisTurn >= 0 ? 'border-green-200/50' : 'border-red-200/50'} shadow-lg`}
                    initial={{ opacity: 0, y: 10 }} 
                    animate={{ opacity: 1, y: 0 }} 
                    transition={{ duration: 0.5, delay: 0.1 }}
                >
                    <p className={`text-base font-bold ${profitThisTurn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {profitThisTurn >= 0 ? '+' : ''} ${profitThisTurn.toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-600 font-medium">This Turn</p>
                </motion.div>
            </div>

            <div className="grid grid-cols-2 gap-2">
                <motion.div 
                    className="p-2 rounded-lg bg-white/10 backdrop-blur-sm border border-white/20 shadow-lg"
                    initial={{ opacity: 0, x: -10 }} 
                    animate={{ opacity: 1, x: 0 }} 
                    transition={{ duration: 0.5, delay: 0.2 }}
                >
                    <p className={`text-base font-bold ${accentColor}`}>{inventory}</p>
                    <p className="text-xs text-gray-600 font-medium">Inventory</p>
                </motion.div>

                <motion.div 
                    className="p-2 rounded-lg bg-white/10 backdrop-blur-sm border border-white/20 shadow-lg"
                    initial={{ opacity: 0, x: 10 }} 
                    animate={{ opacity: 1, x: 0 }} 
                    transition={{ duration: 0.5, delay: 0.3 }}
                >
                    <p className={`text-base font-bold ${accentColor}`}>{projectedDemand}</p>
                    <p className="text-xs text-gray-600 font-medium">Demand</p>
                </motion.div>
            </div>
            
            {children && <div className="mt-1">{children}</div>}
        </div>
    </motion.div>
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
    <motion.div 
        className="w-full h-full bg-white p-3 rounded-lg shadow-xl border border-gray-100"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
    >
        <h2 className="text-sm font-bold mb-1 text-gray-800">Production Over Time</h2>
        <ResponsiveContainer width="100%" height="90%">
            <AreaChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                <defs>
                    <linearGradient id="colorGreen" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
                    </linearGradient>
                    <linearGradient id="colorBlue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                    </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" stroke="#6b7280" tick={{ fontSize: 10 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
                <Tooltip 
                    contentStyle={{ 
                        backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                        fontSize: '11px'
                    }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Area 
                    type="monotone" 
                    dataKey="green" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    fillOpacity={1} 
                    fill="url(#colorGreen)" 
                    name="Green" 
                />
                <Area 
                    type="monotone" 
                    dataKey="blue" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    fillOpacity={1} 
                    fill="url(#colorBlue)" 
                    name="Blue" 
                />
            </AreaChart>
        </ResponsiveContainer>
    </motion.div>
);

interface EventLogProps {
    logs: string[];
}

const EventLog: React.FC<EventLogProps> = ({ logs }) => (
    <motion.div 
        className="w-full h-full bg-gradient-to-br from-gray-900 to-gray-800 text-white p-3 rounded-lg shadow-xl border border-gray-700"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
    >
        <h2 className="text-sm font-bold mb-1 text-gray-200 flex items-center gap-1">
            <span className="text-green-400">▶</span> Event Log
        </h2>
        <div className="overflow-y-auto h-[calc(100%-1.5rem)] font-mono text-xs">
            <AnimatePresence>
                {logs.slice(0, 8).map((log, index) => (
                    <motion.p 
                        key={index} 
                        className="mb-0.5 text-gray-300 flex items-start gap-1"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ duration: 0.3 }}
                    >
                        <span className="text-green-400">$</span>
                        <span className="text-xs">{log}</span>
                    </motion.p>
                ))}
            </AnimatePresence>
        </div>
    </motion.div>
);

function Dashboard() {
    const [logs, setLogs] = useState<string[]>([]);
    const [client, setClient] = useState<W3CWebSocket | null>(null);
    const [isConnected, setIsConnected] = useState(false);

    const [greenTeamData, setGreenTeamData] = useState({ totalProfit: 0, profitThisTurn: 0, inventory: 0, projectedDemand: 0, productionTarget: 0 });
    const [blueTeamData, setBlueTeamData] = useState({ totalProfit: 0, profitThisTurn: 0, inventory: 0, projectedDemand: 0, productionTarget: 0 });
    const [graphData, setGraphData] = useState<GraphData[]>([]);

    const restartGame = () => {
        // Reset all state
        setGreenTeamData({ totalProfit: 0, profitThisTurn: 0, inventory: 0, projectedDemand: 0, productionTarget: 0 });
        setBlueTeamData({ totalProfit: 0, profitThisTurn: 0, inventory: 0, projectedDemand: 0, productionTarget: 0 });
        setGraphData([]);
        setLogs(['Game restarted! Reconnecting...']);
        
        // Close existing connection
        if (client) {
            client.close();
        }
        
        // Create new connection
        setTimeout(() => {
            const newClient = getWebSocketClient();
            setClient(newClient);
            setupWebSocketHandlers(newClient);
        }, 500);
    };

    const setupWebSocketHandlers = (wsClient: W3CWebSocket) => {
        wsClient.onopen = () => {
            setIsConnected(true);
            setLogs(prev => ['Connected to server!', ...prev].slice(0, 10));
        };

        wsClient.onclose = () => {
            setIsConnected(false);
            setLogs(prev => ['Disconnected from server', ...prev].slice(0, 10));
        };

        wsClient.onerror = () => {
            setIsConnected(false);
            setLogs(prev => ['Connection error!', ...prev].slice(0, 10));
        };

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
                    return [...newLogs, ...prevLogs].slice(0, 10); // Keep last 10 logs, new logs at top
                });
            }
        };
    };

    useEffect(() => {
        const wsClient = getWebSocketClient();
        setClient(wsClient);
        setupWebSocketHandlers(wsClient);

        return () => {
            if (wsClient) {
                wsClient.close();
            }
        };
    }, []);
    
    return(
        <div className="dashboard h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-2 overflow-hidden flex flex-col">
            {/* Header with Restart Button - Fixed Height */}
            <div className="flex justify-end mb-2 h-8">
                <motion.button
                    onClick={restartGame}
                    className={`px-3 py-1 rounded-lg font-bold shadow-lg transition-all duration-300 text-xs ${
                        isConnected 
                            ? 'bg-red-500 hover:bg-red-600 text-white' 
                            : 'bg-gray-400 text-gray-200 cursor-not-allowed'
                    }`}
                    whileHover={isConnected ? { scale: 1.05 } : {}}
                    whileTap={isConnected ? { scale: 0.95 } : {}}
                    disabled={!isConnected}
                >
                    🔄 Restart
                </motion.button>
            </div>

            {/* Main Content Grid - Takes remaining space */}
            <div className="flex-1 grid grid-cols-2 gap-2 min-h-0">
                {/* Left Column - Green Team */}
                <div className="flex flex-col min-h-0">
                    <TeamPanel 
                        title="Green Team (Human)"
                        totalProfit={greenTeamData.totalProfit}
                        profitThisTurn={greenTeamData.profitThisTurn}
                        inventory={greenTeamData.inventory}
                        projectedDemand={greenTeamData.projectedDemand}
                        bgColor="bg-gradient-to-br from-green-50 to-emerald-50"
                        textColor="text-green-800"
                        accentColor="text-green-600"
                        team="green"
                    >
                        <Controls client={client} />
                    </TeamPanel>
                </div>

                {/* Right Column - Blue Team */}
                <div className="flex flex-col min-h-0">
                    <TeamPanel
                        title="Blue Team (MAS)"
                        totalProfit={blueTeamData.totalProfit}
                        profitThisTurn={blueTeamData.profitThisTurn}
                        inventory={blueTeamData.inventory}
                        projectedDemand={blueTeamData.projectedDemand}
                        bgColor="bg-gradient-to-br from-blue-50 to-sky-50"
                        textColor="text-blue-800"
                        accentColor="text-blue-600"
                        team="blue"
                    />
                </div>
            </div>

            {/* Bottom Section - Graph and Logs - Fixed Height */}
            <div className="grid grid-cols-2 gap-2 h-48 mt-2">
                <ProductionGraph data={graphData} />
                <EventLog logs={logs} />
            </div>
        </div>
    );
}

export default Dashboard;