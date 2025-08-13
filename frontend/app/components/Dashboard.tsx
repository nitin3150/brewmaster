'use client';

import React, { useEffect, useState, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

// Team Panel Component
interface TeamPanelProps {
    title: string;
    totalProfit: number;
    profitThisTurn: number;
    inventory: number;
    projectedDemand: number;
    actualDemand?: number;
    price: number;
    production: number;
    marketing: number;
    bgColor: string;
    textColor: string;
    accentColor: string;
    icon: string;
    children?: React.ReactNode;
}

const TeamPanel: React.FC<TeamPanelProps> = ({ 
    title, 
    totalProfit = 0, 
    profitThisTurn = 0, 
    inventory = 0, 
    projectedDemand = 0,
    actualDemand,
    price = 0,
    production = 0,
    marketing = 0,
    bgColor, 
    textColor,
    accentColor,
    icon,
    children 
}) => (
    <motion.div 
        className={`${bgColor} p-3 rounded-lg shadow-xl backdrop-blur-sm border border-white/20 relative overflow-hidden`}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
    >
        <h1 className={`text-lg font-bold ${textColor} mb-3 flex items-center gap-2`}>
            <span className="text-2xl">{icon}</span>
            {title}
        </h1>
        
        <div className="space-y-2">
            {/* Profit Row */}
            <div className="grid grid-cols-2 gap-2">
                <div className="p-2 rounded-lg bg-white/20 backdrop-blur-sm">
                    <p className={`text-2xl font-bold ${accentColor}`}>
                        ${(totalProfit || 0).toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-600">Total Profit</p>
                </div>
                <div className={`p-2 rounded-lg ${profitThisTurn >= 0 ? 'bg-green-50/50' : 'bg-red-50/50'}`}>
                    <p className={`text-lg font-bold ${profitThisTurn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {profitThisTurn >= 0 ? '+' : ''} ${(profitThisTurn || 0).toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-600">This Turn</p>
                </div>
            </div>

            {/* Metrics Row */}
            <div className="grid grid-cols-2 gap-2">
                <div className="p-2 rounded-lg bg-white/20">
                    <p className={`text-lg font-bold ${accentColor}`}>{inventory || 0}</p>
                    <p className="text-xs text-gray-600">Inventory</p>
                </div>
                <div className="p-2 rounded-lg bg-white/20">
                    <p className={`text-lg font-bold ${accentColor}`}>{actualDemand || projectedDemand || 0}</p>
                    <p className="text-xs text-gray-600">{actualDemand ? 'Actual Demand' : 'Projected'}</p>
                </div>
            </div>

            {/* Current Decisions */}
            <div className="p-2 rounded-lg bg-white/10 border border-white/20">
                <p className="text-xs font-semibold text-gray-700 mb-1">Current Strategy:</p>
                <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                        <p className="text-gray-600">Price</p>
                        <p className={`font-bold ${accentColor}`}>${price || 0}</p>
                    </div>
                    <div>
                        <p className="text-gray-600">Production</p>
                        <p className={`font-bold ${accentColor}`}>{production || 0}</p>
                    </div>
                    <div>
                        <p className="text-gray-600">Marketing</p>
                        <p className={`font-bold ${accentColor}`}>${marketing || 0}</p>
                    </div>
                </div>
            </div>
            
            {children}
        </div>
    </motion.div>
);

// Controls Component
interface ControlsProps {
    onSubmit: (decisions: {price: number, productionTarget: number, marketingSpend: number}) => void;
}

const Controls: React.FC<ControlsProps> = ({ onSubmit }) => {
    const [price, setPrice] = useState(10);
    const [productionTarget, setProductionTarget] = useState(50);
    const [marketingSpend, setMarketingSpend] = useState(500);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleSubmit = () => {
        setIsSubmitting(true);
        onSubmit({ price, productionTarget, marketingSpend });
        setTimeout(() => setIsSubmitting(false), 1000);
    };

    return (
        <div className="mt-3 p-3 bg-white/30 rounded-lg border border-white/40">
            <h3 className="text-sm font-bold mb-2 text-green-800">Your Controls</h3>
            
            <div className="space-y-2">
                <div className="flex items-center gap-2">
                    <label className="text-xs font-medium w-16 text-green-800">Price</label>
                    <input
                        type="range"
                        min="8"
                        max="15"
                        step="0.5"
                        value={price}
                        onChange={(e) => setPrice(Number(e.target.value))}
                        className="flex-1 h-2"
                    />
                    <span className="text-sm font-bold text-green-600 w-12">${price}</span>
                </div>

                <div className="flex items-center gap-2">
                    <label className="text-xs font-medium w-16 text-green-800">Production</label>
                    <input
                        type="range"
                        min="0"
                        max="200"
                        step="10"
                        value={productionTarget}
                        onChange={(e) => setProductionTarget(Number(e.target.value))}
                        className="flex-1 h-2"
                    />
                    <span className="text-sm font-bold text-green-600 w-12">{productionTarget}</span>
                </div>

                <div className="flex items-center gap-2">
                    <label className="text-xs font-medium w-16 text-green-800">Marketing</label>
                    <input
                        type="range"
                        min="0"
                        max="2000"
                        step="100"
                        value={marketingSpend}
                        onChange={(e) => setMarketingSpend(Number(e.target.value))}
                        className="flex-1 h-2"
                    />
                    <span className="text-sm font-bold text-green-600 w-12">${marketingSpend}</span>
                </div>

                <motion.button
                    onClick={handleSubmit}
                    className={`w-full py-2 px-4 rounded-md font-bold text-sm shadow-md transition-all ${
                        isSubmitting 
                            ? 'bg-green-400 text-white' 
                            : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    {isSubmitting ? '✓ Submitted!' : 'End Turn'}
                </motion.button>
            </div>
        </div>
    );
};

// Type definitions
interface TeamData {
    totalProfit: number;
    profitThisTurn: number;
    inventory: number;
    projectedDemand: number;
    actualDemand?: number;
    price: number;
    production: number;
    marketing: number;
}

// Main Dashboard Component
export default function Dashboard() {
    const [socket, setSocket] = useState<WebSocket | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [currentTurn, setCurrentTurn] = useState(0);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    
    // Compete mode states
    const [isCompeteMode, setIsCompeteMode] = useState(false);
    const [competeDialogOpen, setCompeteDialogOpen] = useState(false);
    const [competeTurns, setCompeteTurns] = useState(10);
    const [competeProgress, setCompeteProgress] = useState(0);
    
    // Team states
    const [greenTeam, setGreenTeam] = useState<TeamData>({
        totalProfit: 100000, profitThisTurn: 0, inventory: 100, 
        projectedDemand: 50, price: 10, production: 50, marketing: 500
    });
    const [blueTeam, setBlueTeam] = useState<TeamData>({
        totalProfit: 100000, profitThisTurn: 0, inventory: 100, 
        projectedDemand: 50, price: 10, production: 50, marketing: 500
    });
    const [purpleTeam, setPurpleTeam] = useState<TeamData>({
        totalProfit: 100000, profitThisTurn: 0, inventory: 100, 
        projectedDemand: 50, price: 10, production: 50, marketing: 500
    });
    
    // Chart data
    const [productionHistory, setProductionHistory] = useState<any[]>([]);

    const startCompeteMode = () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            setIsCompeteMode(true);
            setCompeteProgress(0);
            socket.send(JSON.stringify({ 
                compete: true, 
                turns: competeTurns 
            }));
            setCompeteDialogOpen(false);
        }
    };
    
    const stopCompeteMode = () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            setIsCompeteMode(false);
            socket.send(JSON.stringify({ stopCompete: true }));
        }
    };

    const connectWebSocket = () => {
        try {
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                setIsConnected(true);
                setLogs(prev => ['Connected to server!', ...prev].slice(0, 10));
                if (reconnectTimeoutRef.current) {
                    clearTimeout(reconnectTimeoutRef.current);
                    reconnectTimeoutRef.current = null;
                }
            };

            ws.onclose = () => {
                setIsConnected(false);
                setLogs(prev => ['Disconnected from server', ...prev].slice(0, 10));
                reconnectTimeoutRef.current = setTimeout(() => {
                    connectWebSocket();
                }, 3000);
            };

            ws.onerror = (error) => {
                setIsConnected(false);
                setLogs(prev => ['Connection error!', ...prev].slice(0, 10));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                setCurrentTurn(data.turn || 0);
                
                // Update team states
                setGreenTeam({
                    totalProfit: data.green_team_profit || 100000,
                    profitThisTurn: data.green_team_profit_this_turn || 0,
                    inventory: data.green_team_inventory || 100,
                    projectedDemand: data.green_team_projected_demand || 50,
                    actualDemand: data.green_team_actual_demand,
                    price: data.green_team_price || 10,
                    production: data.green_team_production_target || 50,
                    marketing: data.green_team_marketing_spend || 500
                });
                
                setBlueTeam({
                    totalProfit: data.blue_team_profit || 100000,
                    profitThisTurn: data.blue_team_profit_this_turn || 0,
                    inventory: data.blue_team_inventory || 100,
                    projectedDemand: data.blue_team_projected_demand || 50,
                    actualDemand: data.blue_team_actual_demand,
                    price: data.blue_team_price || 10,
                    production: data.blue_team_production_target || 50,
                    marketing: data.blue_team_marketing_spend || 500
                });
                
                if ('purple_team_profit' in data) {
                    setPurpleTeam({
                        totalProfit: data.purple_team_profit || 100000,
                        profitThisTurn: data.purple_team_profit_this_turn || 0,
                        inventory: data.purple_team_inventory || 100,
                        projectedDemand: data.purple_team_projected_demand || 50,
                        actualDemand: data.purple_team_actual_demand,
                        price: data.purple_team_price || 10,
                        production: data.purple_team_production_target || 50,
                        marketing: data.purple_team_marketing_spend || 500
                    });
                }
                
                // Update production history
                if (data.turn > 0) {
                    setProductionHistory(prev => {
                        const newEntry = {
                            turn: data.turn,
                            'Human': data.green_team_production_target || 0,
                            'Mesa MAS': data.blue_team_production_target || 0,
                            'Temporal MAS': data.purple_team_production_target || 0
                        };
                        return [...prev, newEntry].slice(-20);
                    });
                }
                
                // Update compete progress
                if (data.compete_progress !== undefined) {
                    setCompeteProgress(data.compete_progress);
                    if (data.compete_complete) {
                        setIsCompeteMode(false);
                        setLogs(prev => ['🏆 Competition Complete! Check results above.', ...prev]);
                    }
                }
                
                // Update logs
                const turnLogs = data.event_log || [];
                if (data.turn > 0) {
                    const enhancedLogs = [...turnLogs];
                    enhancedLogs.push(`Human: Price $${data.green_team_price}, Prod ${data.green_team_production_target}, Mkt $${data.green_team_marketing_spend}`);
                    enhancedLogs.push(`Mesa: Price $${data.blue_team_price}, Prod ${data.blue_team_production_target}, Mkt $${data.blue_team_marketing_spend}`);
                    if ('purple_team_profit' in data) {
                        enhancedLogs.push(`Temporal: Price $${data.purple_team_price}, Prod ${data.purple_team_production_target}, Mkt $${data.purple_team_marketing_spend}`);
                    }
                    setLogs(prev => [...enhancedLogs, ...prev].slice(0, 20));
                } else {
                    setLogs(prev => [...turnLogs, ...prev].slice(0, 20));
                }
            };
            
            setSocket(ws);
        } catch (error) {
            console.error('WebSocket connection error:', error);
            setLogs(prev => ['Failed to connect to server', ...prev].slice(0, 10));
        }
    };

    useEffect(() => {
        connectWebSocket();
        
        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (socket) {
                socket.close();
            }
        };
    }, []);

    const handleHumanDecision = (decisions: {price: number, productionTarget: number, marketingSpend: number}) => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(decisions));
        }
    };

    const restartGame = () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ restart: true }));
            setProductionHistory([]);
            setLogs(['Game restarted!']);
            setIsCompeteMode(false);
            setCompeteProgress(0);
        }
    };

    const isThreeTeamMode = logs.some(log => log.includes('Three-way competition')) || 
                           'purple_team_profit' in (greenTeam as any) ||
                           purpleTeam.totalProfit !== 100000;

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4">
            {/* Header */}
            <div className="mb-4 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-800">BrewMasters Competition</h1>
                    <p className="text-gray-600">
                        {isThreeTeamMode ? 'Human vs Mesa MAS vs Temporal MAS' : 'Human vs MAS'} - Turn {currentTurn}
                    </p>
                </div>
                <div className="flex gap-2">
                    <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                        isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                        {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
                    </div>
                    <motion.button
                        onClick={() => setCompeteDialogOpen(true)}
                        className="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white font-bold rounded-lg shadow-md"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        disabled={isCompeteMode}
                    >
                        {isCompeteMode ? '🏃 Competition Running...' : '🏆 Compete Mode'}
                    </motion.button>
                    <motion.button
                        onClick={restartGame}
                        className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg shadow-md"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        🔄 Restart Game
                    </motion.button>
                </div>
            </div>

            {/* Compete Mode Dialog */}
            {competeDialogOpen && (
                <motion.div 
                    className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                >
                    <motion.div 
                        className="bg-white p-6 rounded-lg shadow-xl max-w-md w-full"
                        initial={{ scale: 0.9 }}
                        animate={{ scale: 1 }}
                    >
                        <h2 className="text-xl font-bold mb-4 text-gray-800">AI Competition Mode</h2>
                        <p className="text-gray-600 mb-4">
                            Let the AI teams compete against each other without human intervention.
                        </p>
                        <div className="mb-4">
                            <label className="block text-sm font-medium mb-2 text-gray-700">Number of turns:</label>
                            <input
                                type="number"
                                min="1"
                                max="100"
                                value={competeTurns}
                                onChange={(e) => setCompeteTurns(Math.min(100, Math.max(1, parseInt(e.target.value) || 1)))}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md text-gray-900 bg-white focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                            />
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={startCompeteMode}
                                className="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white font-semibold rounded-md transition-colors"
                            >
                                Start Competition
                            </button>
                            <button
                                onClick={() => setCompeteDialogOpen(false)}
                                className="px-4 py-2 bg-gray-300 hover:bg-gray-400 text-gray-800 font-semibold rounded-md transition-colors"
                            >
                                Cancel
                            </button>
                        </div>
                    </motion.div>
                </motion.div>
            )}

            {/* Competition Progress Bar */}
            {isCompeteMode && (
                <motion.div 
                    className="mb-4 bg-white p-4 rounded-lg shadow-md"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <div className="flex justify-between items-center mb-2">
                        <h3 className="font-bold">Competition Progress</h3>
                        <button
                            onClick={stopCompeteMode}
                            className="px-3 py-1 bg-red-500 hover:bg-red-600 text-white text-sm rounded"
                        >
                            Stop
                        </button>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                        <motion.div 
                            className="bg-purple-500 h-4 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${competeProgress}%` }}
                            transition={{ duration: 0.5 }}
                        />
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                        Turn {Math.round(competeProgress * competeTurns / 100)} of {competeTurns}
                    </p>
                </motion.div>
            )}

            {/* Main Grid */}
            <div className={`grid ${isThreeTeamMode ? 'grid-cols-3' : 'grid-cols-2'} gap-4 mb-4`}>
                {/* Human Panel */}
                <TeamPanel
                    title="Human Player"
                    totalProfit={greenTeam.totalProfit}
                    profitThisTurn={greenTeam.profitThisTurn}
                    inventory={greenTeam.inventory}
                    projectedDemand={greenTeam.projectedDemand}
                    actualDemand={greenTeam.actualDemand}
                    price={greenTeam.price}
                    production={greenTeam.production}
                    marketing={greenTeam.marketing}
                    bgColor="bg-gradient-to-br from-green-50 to-emerald-50"
                    textColor="text-green-800"
                    accentColor="text-green-600"
                    icon="👤"
                >
                    {!isCompeteMode && <Controls onSubmit={handleHumanDecision} />}
                    {isCompeteMode && (
                        <div className="mt-3 p-3 bg-white/30 rounded-lg border border-white/40 text-center">
                            <p className="text-sm text-green-800">AI Competition Mode Active</p>
                            <p className="text-xs text-green-700 mt-1">Human controls disabled</p>
                        </div>
                    )}
                </TeamPanel>

                {/* Mesa MAS Panel */}
                <TeamPanel
                    title="Mesa MAS"
                    totalProfit={blueTeam.totalProfit}
                    profitThisTurn={blueTeam.profitThisTurn}
                    inventory={blueTeam.inventory}
                    projectedDemand={blueTeam.projectedDemand}
                    actualDemand={blueTeam.actualDemand}
                    price={blueTeam.price}
                    production={blueTeam.production}
                    marketing={blueTeam.marketing}
                    bgColor="bg-gradient-to-br from-blue-50 to-sky-50"
                    textColor="text-blue-800"
                    accentColor="text-blue-600"
                    icon="🤖"
                />

                {/* Temporal MAS Panel */}
                {isThreeTeamMode && (
                    <TeamPanel
                        title="Temporal MAS"
                        totalProfit={purpleTeam.totalProfit}
                        profitThisTurn={purpleTeam.profitThisTurn}
                        inventory={purpleTeam.inventory}
                        projectedDemand={purpleTeam.projectedDemand}
                        actualDemand={purpleTeam.actualDemand}
                        price={purpleTeam.price}
                        production={purpleTeam.production}
                        marketing={purpleTeam.marketing}
                        bgColor="bg-gradient-to-br from-purple-50 to-pink-50"
                        textColor="text-purple-800"
                        accentColor="text-purple-600"
                        icon="⚡"
                    />
                )}
            </div>

            {/* Order Volume Chart and Logs Grid */}
            <div className="grid grid-cols-2 gap-4">
                {/* Order Volume Chart */}
                <motion.div 
                    className="bg-white p-4 rounded-lg shadow-xl"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    <h2 className="text-lg font-bold mb-3 text-gray-800">Order Volume (Production Target) Over Time</h2>
                    <p className="text-xs text-gray-600 mb-2">
                        Demonstrates bullwhip effect: MAS's predictive logic vs human reactive decisions
                    </p>
                    <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={productionHistory}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="turn" />
                            <YAxis label={{ value: 'Production Units', angle: -90, position: 'insideLeft' }} />
                            <Tooltip 
                                formatter={(value: any) => [`${value} units`, 'Production']}
                                labelFormatter={(label) => `Turn ${label}`}
                            />
                            <Legend />
                            <Line 
                                type="monotone" 
                                dataKey="Human" 
                                stroke="#10b981" 
                                strokeWidth={2}
                                name="Human (Reactive)"
                                dot={{ r: 4 }}
                            />
                            <Line 
                                type="monotone" 
                                dataKey="Mesa MAS" 
                                stroke="#3b82f6" 
                                strokeWidth={2}
                                name="Mesa MAS (Predictive)"
                                dot={{ r: 4 }}
                            />
                            {isThreeTeamMode && (
                                <Line 
                                    type="monotone" 
                                    dataKey="Temporal MAS" 
                                    stroke="#a855f7" 
                                    strokeWidth={2}
                                    name="Temporal MAS"
                                    dot={{ r: 4 }}
                                />
                            )}
                        </LineChart>
                    </ResponsiveContainer>
                </motion.div>

                {/* Event Log */}
                <motion.div 
                    className="bg-gray-900 text-white p-4 rounded-lg shadow-xl h-full"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                >
                    <h2 className="text-lg font-bold mb-2 text-gray-200">📋 Event Log</h2>
                    <div className="h-[250px] overflow-y-auto font-mono text-sm space-y-1">
                        <AnimatePresence>
                            {logs.map((log, index) => {
                                let textColor = 'text-gray-300';
                                if (log.includes('Human:') || log.includes('Human Actual:')) textColor = 'text-green-400';
                                else if (log.includes('Mesa:') || log.includes('Mesa Actual:')) textColor = 'text-blue-400';
                                else if (log.includes('Temporal:') || log.includes('Temporal Actual:')) textColor = 'text-purple-400';
                                else if (log.includes('Turn')) textColor = 'text-yellow-400';
                                else if (log.includes('Market sentiment:')) textColor = 'text-orange-400';
                                else if (log.includes('🏆')) textColor = 'text-yellow-300';
                                
                                return (
                                    <motion.div
                                        key={`${index}-${log}`}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        exit={{ opacity: 0, x: 20 }}
                                        className={textColor}
                                    >
                                        $ {log}
                                    </motion.div>
                                );
                            })}
                        </AnimatePresence>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}