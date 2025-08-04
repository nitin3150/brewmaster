'use client';
import { useState } from 'react';
import { w3cwebsocket as W3CWebSocket } from 'websocket';
import { motion } from 'framer-motion';

interface ControlsProps {
    client: W3CWebSocket | null;
}

function Controls({ client }: ControlsProps) {
    const [price, setPrice] = useState(10);
    const [productionTarget, setProductionTarget] = useState(50);
    const [marketingSpend, setMarketingSpend] = useState(500);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const endTurn = () => {
        if (client) {
            setIsSubmitting(true);
            const decisions = {
                price,
                productionTarget,
                marketingSpend,
            };
            client.send(JSON.stringify(decisions));
            
            // Reset animation after a delay
            setTimeout(() => setIsSubmitting(false), 1000);
        }
    };

    return (
        <motion.div 
            className="bg-white/20 backdrop-blur-sm p-2 rounded-lg mt-2 border border-white/30 shadow-lg"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
        >
            <h2 className="text-sm font-bold mb-2 text-green-800">Control Panel</h2>
            
            <div className="space-y-2">
                {/* Price Control */}
                <div className="flex items-center gap-2">
                    <label className="text-xs font-medium text-gray-700 w-20">Price</label>
                    <input
                        type="range"
                        min="8"
                        max="15"
                        value={price}
                        onChange={(e) => setPrice(Number(e.target.value))}
                        className="flex-1 h-1.5 bg-gradient-to-r from-green-200 to-green-400 rounded-full appearance-none cursor-pointer slider"
                    />
                    <span className="text-sm font-bold text-green-600 w-10 text-right">${price}</span>
                </div>

                {/* Production Target Control */}
                <div className="flex items-center gap-2">
                    <label className="text-xs font-medium text-gray-700 w-20">Production</label>
                    <input
                        type="range"
                        min="0"
                        max="1000"
                        value={productionTarget}
                        onChange={(e) => setProductionTarget(Number(e.target.value))}
                        className="flex-1 h-1.5 bg-gradient-to-r from-green-200 to-green-400 rounded-full appearance-none cursor-pointer slider"
                    />
                    <span className="text-sm font-bold text-green-600 w-10 text-right">{productionTarget}</span>
                </div>

                {/* Marketing Spend Control */}
                <div className="flex items-center gap-2">
                    <label className="text-xs font-medium text-gray-700 w-20">Marketing</label>
                    <input
                        type="range"
                        min="0"
                        max="2000"
                        step="100"
                        value={marketingSpend}
                        onChange={(e) => setMarketingSpend(Number(e.target.value))}
                        className="flex-1 h-1.5 bg-gradient-to-r from-green-200 to-green-400 rounded-full appearance-none cursor-pointer slider"
                    />
                    <span className="text-sm font-bold text-green-600 w-10 text-right">${marketingSpend}</span>
                </div>

                {/* Submit Button */}
                <motion.button
                    onClick={endTurn}
                    className={`w-full py-1.5 px-3 rounded-md font-bold text-xs shadow-md transition-all duration-300 ${
                        isSubmitting 
                            ? 'bg-green-400 text-white' 
                            : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    animate={isSubmitting ? { scale: [1, 1.05, 1] } : {}}
                >
                    {isSubmitting ? '✓ Submitted!' : 'End Turn'}
                </motion.button>
            </div>

            <style jsx>{`
                .slider::-webkit-slider-thumb {
                    appearance: none;
                    width: 12px;
                    height: 12px;
                    background: white;
                    border: 2px solid #10b981;
                    border-radius: 50%;
                    cursor: pointer;
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                    transition: all 0.2s;
                }
                
                .slider::-webkit-slider-thumb:hover {
                    transform: scale(1.2);
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                }
                
                .slider::-moz-range-thumb {
                    width: 12px;
                    height: 12px;
                    background: white;
                    border: 2px solid #10b981;
                    border-radius: 50%;
                    cursor: pointer;
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                    transition: all 0.2s;
                }
                
                .slider::-moz-range-thumb:hover {
                    transform: scale(1.2);
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                }
            `}</style>
        </motion.div>
    );
};

export default Controls;