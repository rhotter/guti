import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { KatexBlock } from '../components/KatexBlock';

const ChannelCodingPage = () => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    // Generate data for visualization
    const generateData = () => {
      // Number of frequency components
      const N = 64;
      
      // Create a signal with specific frequency components
      const signal = [];
      for (let i = 0; i < N; i++) {
        signal.push(
          3 * Math.sin(2 * Math.PI * i / N * 2) + 
          2 * Math.sin(2 * Math.PI * i / N * 5) +
          Math.sin(2 * Math.PI * i / N * 8)
        );
      }
      
      // Represent power allocation across frequencies
      const frequencies = Array.from({ length: N }, (_, i) => i);
      
      // Simplified Shannon capacity calculation for each frequency
      const calculateCapacities = (snr) => {
        return frequencies.map(f => {
          // Model frequency-dependent SNR (lower for higher frequencies)
          const freqFactor = Math.exp(-f / 20);
          const effectiveSNR = snr * freqFactor;
          
          // Shannon capacity formula: C = log2(1 + SNR)
          return Math.log2(1 + effectiveSNR);
        });
      };
      
      // Generate capacity curves for different SNR values
      const lowSNR = calculateCapacities(1);
      const mediumSNR = calculateCapacities(10);
      const highSNR = calculateCapacities(100);
      
      // Simulate water-filling power allocation
      const waterFilling = (snr, noiseProfile) => {
        const N = noiseProfile.length;
        const inverseNoise = noiseProfile.map(n => 1 / n);
        const sortedIndices = Array.from({ length: N }, (_, i) => i)
          .sort((a, b) => inverseNoise[b] - inverseNoise[a]);
        
        // Simplified water-filling algorithm
        const powerAllocation = Array(N).fill(0);
        let remainingPower = snr * N;
        let numChannels = 0;
        
        while (numChannels < N && remainingPower > 0) {
          numChannels++;
          
          // Calculate water level
          const waterLevel = remainingPower / numChannels + 
            sortedIndices.slice(0, numChannels).reduce((sum, idx) => sum + noiseProfile[idx], 0) / numChannels;
          
          // Allocate power based on water level
          let totalAllocated = 0;
          for (let i = 0; i < numChannels; i++) {
            const idx = sortedIndices[i];
            const power = Math.max(0, waterLevel - noiseProfile[idx]);
            powerAllocation[idx] = power;
            totalAllocated += power;
          }
          
          // Check if we've allocated all power
          if (totalAllocated <= remainingPower) {
            remainingPower = 0;
          } else {
            // Back up and try with fewer channels
            numChannels--;
            if (numChannels === 0) break;
          }
        }
        
        return powerAllocation;
      };
      
      // Generate noise profile (higher noise at higher frequencies)
      const noiseProfile = frequencies.map(f => 0.1 + 0.01 * f);
      
      // Calculate power allocation using water-filling
      const powerAllocation = waterFilling(10, noiseProfile);
      
      // Calculate the resulting capacity with this power allocation
      const optimizedCapacity = frequencies.map((f, i) => {
        const snr = powerAllocation[i] / noiseProfile[i];
        return Math.log2(1 + snr);
      });
      
      return {
        frequencies,
        lowSNR,
        mediumSNR,
        highSNR,
        noiseProfile,
        powerAllocation,
        optimizedCapacity
      };
    };

    setData(generateData());
  }, []);

  if (!data) {
    return <div>Loading visualization...</div>;
  }

  return (
    <div className="page-container">
      <h1>SNR and Channel Coding in Fourier Space</h1>
      
      <section className="explanation">
        <h2>Channel Capacity in Frequency Domain</h2>
        <p>
          In communication systems, the Shannon-Hartley theorem shows that the channel capacity 
          depends on the signal-to-noise ratio (SNR). When we transform a signal to the Fourier domain, 
          each frequency component can be treated as an independent channel with its own SNR.
        </p>
        
        <KatexBlock math="C = \sum_{i=1}^{N} \log_2\left(1 + \frac{P_i}{N_i}\right)" />
        
        <p>
          Where C is the total capacity, P_i is the power allocated to the ith frequency component, and 
          N_i is the noise power at that frequency. The optimal power allocation follows the water-filling 
          principle, allocating more power to frequencies with better SNR.
        </p>
      </section>

      <section className="visualization">
        <h2>Visualization</h2>
        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.frequencies.slice(0, 32),
                y: data.lowSNR.slice(0, 32),
                type: 'scatter',
                mode: 'lines',
                name: 'Low SNR (0dB)'
              },
              {
                x: data.frequencies.slice(0, 32),
                y: data.mediumSNR.slice(0, 32),
                type: 'scatter',
                mode: 'lines',
                name: 'Medium SNR (10dB)'
              },
              {
                x: data.frequencies.slice(0, 32),
                y: data.highSNR.slice(0, 32),
                type: 'scatter',
                mode: 'lines',
                name: 'High SNR (20dB)'
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Channel Capacity per Frequency (bits/use)',
              xaxis: { title: 'Frequency' },
              yaxis: { title: 'Capacity (bits/use)' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.frequencies.slice(0, 32),
                y: data.noiseProfile.slice(0, 32),
                type: 'scatter',
                mode: 'lines',
                name: 'Noise Profile',
                line: { color: 'red' }
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Noise Profile Across Frequencies',
              xaxis: { title: 'Frequency' },
              yaxis: { title: 'Noise Power' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.frequencies.slice(0, 32),
                y: data.powerAllocation.slice(0, 32),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Power Allocation',
                line: { color: 'green' }
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Water-Filling Power Allocation',
              xaxis: { title: 'Frequency' },
              yaxis: { title: 'Power Allocated' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.frequencies.slice(0, 32),
                y: data.optimizedCapacity.slice(0, 32),
                type: 'scatter',
                mode: 'lines',
                name: 'Optimized Capacity',
                line: { color: 'purple' }
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Capacity with Optimized Power Allocation',
              xaxis: { title: 'Frequency' },
              yaxis: { title: 'Capacity (bits/use)' }
            }}
          />
        </div>
      </section>

      <section className="future-work">
        <h3>Interactive Elements to be Added:</h3>
        <ul>
          <li>Adjustable total power budget</li>
          <li>Customizable noise profile</li>
          <li>Comparison between different power allocation strategies</li>
          <li>Visualization of how water-filling algorithm works</li>
          <li>Computation of total capacity and efficiency metrics</li>
        </ul>
      </section>
    </div>
  );
};

export default ChannelCodingPage; 