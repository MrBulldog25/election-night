import React, { useState, useEffect } from 'react';
import NJMap from './components/NJMap';
import Dashboard from './components/Dashboard';
import { fetchInitialData, updateModel, resetModel } from './api';

import HypotheticalsPanel from './components/HypotheticalsPanel';

function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [geojson, setGeojson] = useState(null);
  const [stats, setStats] = useState(null);
  const [countyMetadata, setCountyMetadata] = useState([]);
  const [appliedObservations, setAppliedObservations] = useState([]);
  const [realResults, setRealResults] = useState([]);

  const [selectedCounty, setSelectedCounty] = useState(null);
  const [stagedObservations, setStagedObservations] = useState([]);

  useEffect(() => {
    const init = async () => {
      try {
        const data = await fetchInitialData();
        setGeojson(data.geojson);
        setStats(data.stats);
        setCountyMetadata(data.county_metadata);
        setAppliedObservations(data.applied_observations);
        setRealResults(data.real_results || []);
      } catch (err) {
        console.error(err);
        setError("Failed to load initial data.");
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  const handleCountyClick = (county) => {
    if (selectedCounty === county) {
      setSelectedCounty(null);
    } else {
      setSelectedCounty(county);
    }
  };

  const handleStageObservation = (obs) => {
    setStagedObservations([...stagedObservations, obs]);
  };

  const handleRemoveStaged = (idx) => {
    const newStaged = [...stagedObservations];
    newStaged.splice(idx, 1);
    setStagedObservations(newStaged);
  };

  const handleApply = async () => {
    try {
      const allObs = [...appliedObservations, ...stagedObservations];
      const data = await updateModel(allObs);
      setStats(data.stats);
      setAppliedObservations(data.applied_observations);
      setRealResults(data.real_results || realResults);
      setStagedObservations([]);
    } catch (err) {
      console.error(err);
      alert("Failed to apply updates.");
    }
  };

  const handleReset = async () => {
    try {
      const data = await resetModel();
      setStats(data.stats);
      setAppliedObservations(data.applied_observations);
      setRealResults(data.real_results || []);
      setStagedObservations([]);
      setSelectedCounty(null);
    } catch (err) {
      console.error(err);
      alert("Failed to reset model.");
    }
  };

  if (loading) return <div className="flex items-center justify-center h-screen text-gray-500">Loading application...</div>;
  if (error) return <div className="flex items-center justify-center h-screen text-red-500">{error}</div>;

  return (
    <div className="flex h-screen w-screen overflow-hidden font-sans bg-slate-50">
      <div className="w-[500px] h-full flex-shrink-0 border-r border-slate-200 bg-white z-10 shadow-xl flex flex-col">
        <Dashboard
          stats={stats}
          selectedCounty={selectedCounty}
          countyMetadata={countyMetadata}
          onStageObservation={handleStageObservation}
          stagedObservations={stagedObservations}
          onRemoveStaged={handleRemoveStaged}
          onApply={handleApply}
          onReset={handleReset}
          appliedObservations={appliedObservations}
          realResults={realResults}
        />
      </div>

      <div className="flex-grow flex flex-col relative bg-slate-100">
        <div className="flex-grow relative">
          <div className="absolute inset-0">
            <NJMap
              geojson={geojson}
              stats={stats}
              countyMetadata={countyMetadata}
              onCountyClick={handleCountyClick}
              selectedCounty={selectedCounty}
              realResults={realResults}
            />
          </div>
        </div>

        <div className="h-48 flex-shrink-0 z-10">
          <HypotheticalsPanel appliedObservations={appliedObservations} />
        </div>
      </div>
    </div>
  );
}

export default App;
