import React from 'react';

const formatPercent = (val) => {
    if (val === undefined || val === null) return '-';
    return `${(val * 100).toFixed(1)}%`;
};

const HypotheticalsPanel = ({ appliedObservations }) => {
    if (!appliedObservations || appliedObservations.length === 0) {
        return (
            <div className="h-full flex flex-col items-center justify-center text-center p-6 bg-white border-t border-slate-200">
                <p className="text-slate-400 text-sm font-medium">No active hypotheticals.</p>
            </div>
        );
    }

    return (
        <div className="h-full bg-white border-t border-slate-200 flex flex-col">
            <div className="px-6 py-3 border-b border-slate-100 bg-slate-50/50">
                <h3 className="text-sm font-bold text-slate-900">Current Hypotheticals</h3>
            </div>
            <div className="flex-grow overflow-y-auto p-6">
                <ul className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {appliedObservations.map((obs, idx) => (
                        <li key={idx} className="bg-slate-50 border border-slate-200 p-3 rounded-lg text-sm flex justify-between items-center shadow-sm">
                            <div className="flex flex-col">
                                <span className="font-semibold text-slate-900">{obs.county}</span>
                                <span className="text-xs text-slate-500">
                                    {obs.kind === 'votes' && `Votes (${formatPercent(obs.share)})`}
                                    {obs.kind === 'share' && `Share ${formatPercent(obs.share)}`}
                                    {obs.kind === 'winner' && `Winner: ${obs.winner === 'dem' ? 'Dem' : 'Rep'}`}
                                </span>
                            </div>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default HypotheticalsPanel;
