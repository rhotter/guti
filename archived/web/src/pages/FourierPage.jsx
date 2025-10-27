import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { KatexBlock } from '../components/KatexBlock';

const FourierPage = () => {
  const [data, setData] = useState([]);
  
  useEffect(() => {
    // Data preparation function
    const generateData = () => {
      // Create a simple signal
      const N = 128; // Number of points, power of 2 for efficient FFT
      const t = Array.from({ length: N }, (_, i) => i);
      
      // Create a simple signal: sum of two sine waves with noise
      const signal = t.map(ti => 
        Math.sin(2 * Math.PI * ti / 20) + 
        0.5 * Math.sin(2 * Math.PI * ti / 5) + 
        0.3 * (Math.random() - 0.5)
      );

      // Calculate the Fourier transform
      // For simplicity, we're using a function that mimics FFT behavior
      const fftMagnitude = calculateFFTMagnitude(signal);
      
      // Simple Gaussian filter in frequency domain
      const filter = fftMagnitude.map((_, i) => 
        Math.exp(-Math.pow((i - N/2) / (N/10), 2))
      );
      
      // Filter the signal in frequency domain (conceptual)
      const filteredMagnitude = fftMagnitude.map((val, i) => val * filter[i]);
      
      return {
        time: t,
        signal,
        frequencies: Array.from({ length: N }, (_, i) => i - N/2),
        fftMagnitude,
        filter,
        filteredMagnitude
      };
    };

    // Simple function to calculate the magnitude of the FFT
    // In a real implementation, we'd use a proper FFT algorithm
    const calculateFFTMagnitude = (signal) => {
      const N = signal.length;
      const result = [];
      
      for (let k = 0; k < N; k++) {
        let real = 0;
        let imag = 0;
        
        for (let n = 0; n < N; n++) {
          const angle = -2 * Math.PI * k * n / N;
          real += signal[n] * Math.cos(angle);
          imag += signal[n] * Math.sin(angle);
        }
        
        // Normalize and shift to center the zero frequency
        result.push(Math.sqrt(real*real + imag*imag) / N);
      }
      
      // Shift the result to center the zero frequency
      const firstHalf = result.slice(N/2);
      const secondHalf = result.slice(0, N/2);
      return [...firstHalf, ...secondHalf];
    };

    setData(generateData());
  }, []);

  // Render loading state if data is not yet available
  if (data.length === 0) {
    return <div>Loading visualization...</div>;
  }

  return (
    <div className="page-container">
      <h1>Fourier Transform Visualization</h1>
      
      <section className="explanation">
        <h2>Linear Models in Fourier Space</h2>
        <p>
          Linear operations like smoothing can be understood as multiplications in the Fourier domain. 
          The Fourier transform converts a signal from the time/space domain to the frequency domain.
        </p>
        
        <KatexBlock math="F(y) = F(Ax) = F(A) \cdot F(x)" />
        
        <p>
          This shows that applying a linear operator A to a signal x and then taking the Fourier transform is 
          equivalent to taking the Fourier transform of both and multiplying them in the frequency domain.
        </p>
      </section>

      <section className="visualization">
        <h2>Visualization</h2>
        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.time,
                y: data.signal,
                type: 'scatter',
                mode: 'lines',
                name: 'Original Signal'
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Time Domain Signal',
              xaxis: { title: 'Time' },
              yaxis: { title: 'Amplitude' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.frequencies,
                y: data.fftMagnitude,
                type: 'scatter',
                mode: 'lines',
                name: 'Frequency Spectrum'
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Frequency Domain Representation',
              xaxis: { title: 'Frequency' },
              yaxis: { title: 'Magnitude' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.frequencies,
                y: data.filter,
                type: 'scatter',
                mode: 'lines',
                name: 'Gaussian Filter',
                line: { color: 'red' }
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Frequency Domain Filter',
              xaxis: { title: 'Frequency' },
              yaxis: { title: 'Filter Response' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.frequencies,
                y: data.filteredMagnitude,
                type: 'scatter',
                mode: 'lines',
                name: 'Filtered Spectrum',
                line: { color: 'green' }
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Filtered Frequency Spectrum',
              xaxis: { title: 'Frequency' },
              yaxis: { title: 'Magnitude' }
            }}
          />
        </div>
      </section>

      <section className="future-work">
        <h3>Interactive Elements to be Added:</h3>
        <ul>
          <li>Interactive filter design in the frequency domain</li>
          <li>Real-time visualization of filtering effects</li>
          <li>Side-by-side comparison of time domain filtering vs. frequency domain filtering</li>
        </ul>
      </section>
    </div>
  );
};

export default FourierPage; 