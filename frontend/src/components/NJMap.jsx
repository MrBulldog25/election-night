import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

const NJMap = ({ geojson, stats, countyMetadata, onCountyClick, selectedCounty, realResults }) => {
  const pDemWin = (countyName) => {
    const s = stats?.counties?.[countyName];
    if (!s) return 0.5;
    if (typeof s.p_dem_win === 'number') return s.p_dem_win;
    return s.dem_share ? s.dem_share.mean : 0.5;
  };

  const realLookup = useMemo(() => {
    const map = new Map();
    (realResults || []).forEach((r) => {
      const denom = Math.max(r.dem_votes + r.rep_votes, 1);
      const share = r.dem_votes / denom;
      map.set(r.county, { ...r, share });
    });
    return map;
  }, [realResults]);

  const data = useMemo(() => {
    if (!geojson || !countyMetadata) return [];

    const locations = countyMetadata.map(c => c.geo_id);
    const z = countyMetadata.map((c) => pDemWin(c.county));

    const hovertext = countyMetadata.map(c => {
      const countyName = c.county;
      const real = realLookup.get(countyName);
      if (real) {
        const totalTwoParty = real.dem_votes + real.rep_votes;
        return `<b>${c.county_clean} County</b><br>` +
          `Live Dem Share: ${(real.share * 100).toFixed(1)}%<br>` +
          `Reporting: ${real.reporting_pct}%<br>` +
          `Dem: ${real.dem_votes.toLocaleString()} | Rep: ${real.rep_votes.toLocaleString()}<br>` +
          `Total (2p): ${totalTwoParty.toLocaleString()}`;
      }
      const s = stats?.counties?.[countyName];
      if (!s) return c.county_clean;

      return `<b>${c.county_clean} County</b><br>` +
        `Dem Win Odds: ${(pDemWin(countyName) * 100).toFixed(1)}%<br>` +
        `Dem Expected Share: ${(s.dem_share.mean * 100).toFixed(1)}%<br>` +
        `Total Votes: ${Math.round(Number(s.total_votes.mean)).toLocaleString()}`;
    });

    const customdata = countyMetadata.map(c => ({
      county: c.county,
      clean: c.county_clean,
      geo_id: c.geo_id
    }));

    const colorscale = [
      [0.0, '#7f1d1d'],
      [0.25, '#fca5a5'],
      [0.5, '#e2e8f0'],
      [0.75, '#bfdbfe'],
      [1.0, '#1d4ed8'],
    ];

    return [{
      type: 'choropleth',
      geojson: geojson,
      locations: locations,
      z: z,
      featureidkey: 'properties.GEO_ID',
      colorscale: colorscale,
      zmin: 0.0,
      zmax: 1.0,
      marker: { line: { width: 0.5, color: '#94A3B8' } },
      showscale: false,
      hoverinfo: 'text',
      hovertext: hovertext,
      customdata: customdata,
      selectedpoints: selectedCounty ? [countyMetadata.findIndex(c => c.county === selectedCounty)] : null,
      selected: { marker: { opacity: 1, line: { width: 2, color: '#0F172A' } } },
      unselected: { marker: { opacity: 0.5 } },
    }];
  }, [geojson, stats, countyMetadata, selectedCounty, realLookup]);

  const layout = useMemo(() => ({
    margin: { l: 0, r: 0, t: 0, b: 0 },
    height: 700,
    geo: {
      fitbounds: 'locations',
      visible: false,
    },
    dragmode: "pan",
    hoverlabel: {
      bgcolor: "white",
      bordercolor: "#CBD5E1",
      font: { family: "Inter, sans-serif", size: 13, color: "#0F172A" }
    }
  }), []);

  const handleClick = (event) => {
    const point = event.points[0];
    if (point && point.customdata) {
      onCountyClick(point.customdata.county);
    }
  };

  return (
    <div className="w-full h-full">
      <Plot
        data={data}
        layout={layout}
        useResizeHandler={true}
        className="w-full h-full"
        onClick={handleClick}
        config={{ displayModeBar: false, scrollZoom: true }}
      />
    </div>
  );
};

export default NJMap;
