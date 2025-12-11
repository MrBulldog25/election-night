import React, { useState, useEffect } from 'react';

const formatPercent = (val) => {
    if (val === undefined || val === null) return '-';
    return `${(val * 100).toFixed(1)}%`;
};

const formatInt = (val) => {
    if (val === undefined || val === null) return '-';
    const rounded = Math.round(Number(val));
    if (!Number.isFinite(rounded)) return '-';
    return rounded.toLocaleString();
};

const Dashboard = ({
    stats,
    selectedCounty,
    countyMetadata,
    onStageObservation,
    stagedObservations,
    onRemoveStaged,
    onApply,
    onReset,
    appliedObservations,
    realResults,
}) => {
    const [demShare, setDemShare] = useState('');
    const [totalVotes, setTotalVotes] = useState('');
    const [winner, setWinner] = useState('');
    const [showResetConfirm, setShowResetConfirm] = useState(false);

    useEffect(() => {
        setDemShare('');
        setTotalVotes('');
        setWinner('');
    }, [selectedCounty]);

    if (!stats) return <div className="p-8 text-slate-500 font-medium">Loading stats...</div>;

    const statewide = stats.statewide;
    const countyStats = selectedCounty && stats.counties ? stats.counties[selectedCounty] : null;
    const countyReal = selectedCounty
        ? (realResults || []).find(r => r.county === selectedCounty)
        : null;

    const handleStage = (e) => {
        e.preventDefault();
        if (!selectedCounty) return;

        const share = demShare ? parseFloat(demShare) / 100 : null;
        const total = totalVotes ? parseInt(totalVotes.replace(/,/g, '')) : null;
        const win = winner || null;

        if (share === null && total === null && win === null) return;

        let obs = { county: selectedCounty };

        if (share !== null && total !== null) {
            const dem = Math.round(share * total);
            const rep = total - dem;
            obs = { ...obs, kind: 'votes', dem_votes: dem, rep_votes: rep, share, total };
        } else if (share !== null) {
            obs = { ...obs, kind: 'share', share };
        } else if (win !== null) {
            obs = { ...obs, kind: 'winner', winner: win };
        } else {
            alert("Please provide at least a share percentage or a winner constraint.");
            return;
        }

        onStageObservation(obs);
        setDemShare('');
        setTotalVotes('');
        setWinner('');
    };

    return (
        <div className="flex flex-col h-full bg-white overflow-hidden font-sans">
            <div className="p-6 border-b border-slate-100 bg-white flex-shrink-0">
                <div className="flex justify-between items-start">
                    <div>
                        <h2 className="text-2xl font-black text-slate-900 tracking-tight mb-1">NJ Governor Election</h2>
                    </div>
                    <div className="flex gap-2">
                        <button
                            onClick={onApply}
                            disabled={stagedObservations.length === 0}
                            className={`font-semibold py-2 px-4 rounded-lg shadow-sm transition transform active:scale-[0.99] text-sm ${stagedObservations.length > 0
                                ? 'bg-slate-900 hover:bg-slate-800 text-white'
                                : 'bg-slate-100 text-slate-400 cursor-not-allowed'
                                }`}
                        >
                            Apply Updates
                        </button>
                        {showResetConfirm ? (
                            <div className="flex gap-2">
                                <button
                                    onClick={() => setShowResetConfirm(false)}
                                    className="bg-white hover:bg-slate-50 text-slate-600 font-medium py-2 px-3 border border-slate-200 rounded-lg text-sm"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={() => {
                                        onReset();
                                        setShowResetConfirm(false);
                                    }}
                                    className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-3 rounded-lg text-sm"
                                >
                                    Confirm
                                </button>
                            </div>
                        ) : (
                            <button
                                onClick={() => setShowResetConfirm(true)}
                                className="bg-white hover:bg-slate-50 text-slate-700 font-medium py-2 px-4 border border-slate-200 rounded-lg transition hover:border-slate-300 text-sm"
                            >
                                Reset
                            </button>
                        )}
                    </div>
                </div>
            </div>

            <div className="flex-grow overflow-y-auto p-6 space-y-6">

                <div className="border border-slate-200 rounded-xl overflow-hidden bg-white">
                    <div className="p-4 border-b border-slate-100 bg-slate-50/50">
                        <h3 className="text-sm font-bold text-slate-900">Projected Vote Count</h3>
                    </div>

                    <div className="p-4 space-y-3">
                        <div className="flex justify-between text-sm font-semibold text-slate-700">
                            <span>Democrat</span>
                            <span className="text-slate-600">Republican</span>
                        </div>
                        <div className="h-6 w-full bg-slate-200 rounded-md overflow-hidden flex">
                            <div
                                className="h-full bg-blue-600"
                                style={{ width: `${(statewide.dem_votes.mean / Math.max(statewide.dem_votes.mean + statewide.rep_votes.mean, 1e-9)) * 100}%` }}
                                title="Projected Democratic vote count"
                            ></div>
                            <div
                                className="h-full bg-red-600"
                                style={{ width: `${(statewide.rep_votes.mean / Math.max(statewide.dem_votes.mean + statewide.rep_votes.mean, 1e-9)) * 100}%` }}
                                title="Projected Republican vote count"
                            ></div>
                        </div>
                        <div className="flex justify-between text-xs text-slate-600">
                            <span>
                                {formatInt(statewide.dem_votes.mean)} votes (
                                {formatPercent(
                                    statewide.dem_votes.mean /
                                    Math.max(statewide.dem_votes.mean + statewide.rep_votes.mean, 1e-9)
                                )}
                                )
                            </span>
                            <span>
                                {formatInt(statewide.rep_votes.mean)} votes (
                                {formatPercent(
                                    statewide.rep_votes.mean /
                                    Math.max(statewide.dem_votes.mean + statewide.rep_votes.mean, 1e-9)
                                )}
                                )
                            </span>
                        </div>
                        <div className="text-sm text-slate-700">
                            {(() => {
                                const demWin = statewide.p_dem_win;
                                const repWin = statewide.p_rep_win;
                                const winner = demWin >= repWin ? "Democrat" : "Republican";
                                const odds = demWin >= repWin ? demWin : repWin;
                                return `The ${winner} has a ${formatPercent(odds)} chance of winning.`;
                            })()}
                        </div>
                    </div>
                </div>

                {selectedCounty ? (
                    <div className="animate-in fade-in slide-in-from-right-4 duration-300">
                        <div className="border border-slate-200 rounded-xl overflow-hidden mb-6">
                            <div className="bg-slate-100 px-4 py-3 border-b border-slate-200 flex justify-between items-center">
                                <h3 className="font-bold text-slate-800">Selected: {selectedCounty} County</h3>
                                {countyReal && (
                                    <span className="text-[10px] font-bold px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded-full uppercase tracking-wide">
                                        Live {countyReal.reporting_pct}%
                                    </span>
                                )}
                            </div>

                            {countyStats && (
                                <div className="p-4 bg-white">
                                    <div className="flex justify-between items-center mb-4">
                                        <div>
                                            <div className="text-sm text-slate-500 mb-1">Dem Share</div>
                                            <div className="text-xl font-bold text-slate-900">{formatPercent(countyStats.dem_share.mean)}</div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-sm text-slate-500 mb-1">Total Votes</div>
                                            <div className="text-xl font-bold text-slate-900">{formatInt(countyStats.total_votes.mean)}</div>
                                        </div>
                                    </div>

                                    <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden flex mb-2">
                                        <div className="h-full bg-blue-500" style={{ width: `${countyStats.dem_share.mean * 100}%` }}></div>
                                        <div className="h-full bg-red-500" style={{ width: `${countyStats.rep_share.mean * 100}%` }}></div>
                                    </div>

                                    {countyReal && (
                                        <div className="pt-3 mt-3 border-t border-dashed border-slate-100 text-xs text-slate-500 flex justify-between">
                                            <span>Live: {formatPercent(countyReal.dem_votes / Math.max(countyReal.dem_votes + countyReal.rep_votes, 1))} Dem</span>
                                            <span>{countyReal.reporting_pct}% Reporting</span>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        <div className="mb-8">
                            <h4 className="text-sm font-bold text-slate-900 mb-3">Add Hypothetical Result</h4>
                            <form onSubmit={handleStage} className="space-y-3">
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">Dem Share (%)</label>
                                        <input
                                            type="number"
                                            step="0.1"
                                            className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-sm text-slate-900 placeholder-slate-400"
                                            placeholder="e.g. 55.5"
                                            value={demShare}
                                            onChange={(e) => setDemShare(e.target.value)}
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">Total Votes</label>
                                        <input
                                            type="number"
                                            className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-sm text-slate-900 placeholder-slate-400"
                                            placeholder="e.g. Votes"
                                            value={totalVotes}
                                            onChange={(e) => setTotalVotes(e.target.value)}
                                        />
                                    </div>
                                </div>
                                <div>
                                    <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">Winner Constraint</label>
                                    <select
                                        className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-sm text-slate-900"
                                        value={winner}
                                        onChange={(e) => setWinner(e.target.value)}
                                    >
                                        <option value="">None</option>
                                        <option value="dem">Democrat</option>
                                        <option value="rep">Republican</option>
                                    </select>
                                </div>
                                <button
                                    type="submit"
                                    className="w-full bg-slate-200 hover:bg-slate-300 text-slate-800 font-semibold py-2.5 px-4 rounded-lg transition shadow-sm active:scale-[0.99] text-sm"
                                >
                                    Add Hypothetical
                                </button>
                            </form>
                        </div>
                    </div>
                ) : (
                    <div className="h-40 flex flex-col items-center justify-center text-center p-6 border-2 border-dashed border-slate-200 rounded-xl bg-slate-50/50 mb-6">
                        <p className="text-slate-400 text-sm font-medium">Select a county on the map to view details.</p>
                    </div>
                )}

                {stagedObservations.length > 0 && (
                    <div className="mb-6 animate-in slide-in-from-bottom-2 duration-300">
                        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Staged Updates</h4>
                        <ul className="space-y-2">
                            {stagedObservations.map((obs, idx) => (
                                <li key={idx} className="bg-amber-50 border border-amber-100 p-3 rounded-lg text-sm flex justify-between items-center group">
                                    <div className="flex flex-col">
                                        <span className="font-semibold text-slate-900">{obs.county}</span>
                                        <span className="text-xs text-slate-500">
                                            {obs.kind === 'votes' && `Votes (${formatPercent(obs.share)})`}
                                            {obs.kind === 'share' && `Share ${formatPercent(obs.share)}`}
                                            {obs.kind === 'winner' && `Winner: ${obs.winner === 'dem' ? 'Dem' : 'Rep'}`}
                                        </span>
                                    </div>
                                    <button
                                        onClick={() => onRemoveStaged(idx)}
                                        className="text-slate-400 hover:text-red-500 p-1.5 rounded-md hover:bg-red-50 transition"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                                    </button>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Dashboard;
