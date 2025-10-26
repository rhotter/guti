import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { KatexBlock } from '../components/KatexBlock';

const OrthogonalityPage = () => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    // Generate data for visualization
    const generateData = () => {
      // Size of our vectors
      const N = 64;
      
      // Create a signal with a few clear components
      const signal = [];
      for (let i = 0; i < N; i++) {
        signal.push(
          3 * Math.sin(2 * Math.PI * i / N * 2) + 
          2 * Math.sin(2 * Math.PI * i / N * 5) +
          Math.sin(2 * Math.PI * i / N * 8)
        );
      }
      
      // Add some noise
      const noiseLevel = 0.5;
      const noisySignal = signal.map(val => val + noiseLevel * (Math.random() * 2 - 1));
      
      // Simple DFT function (for educational purposes)
      const dft = (signal) => {
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
          
          // Store complex number as [real, imag]
          result.push([real, imag]);
        }
        
        return result;
      };
      
      // Calculate DFT
      const fourierCoefficients = dft(signal);
      const noisyFourierCoefficients = dft(noisySignal);
      
      // Extract magnitudes and phases
      const magnitudes = fourierCoefficients.map(([re, im]) => 
        Math.sqrt(re*re + im*im) / N);
      
      const noisyMagnitudes = noisyFourierCoefficients.map(([re, im]) => 
        Math.sqrt(re*re + im*im) / N);
      
      // Calculate noise difference in Fourier domain
      const fourierDifference = fourierCoefficients.map(([re1, im1], i) => {
        const [re2, im2] = noisyFourierCoefficients[i];
        return [re2 - re1, im2 - im1];
      });
      
      const fourierNoiseMagnitudes = fourierDifference.map(([re, im]) => 
        Math.sqrt(re*re + im*im) / N);
      
      return {
        timeIndices: Array.from({ length: N }, (_, i) => i),
        signal,
        noisySignal,
        frequencyIndices: Array.from({ length: N }, (_, i) => i),
        magnitudes,
        noisyMagnitudes,
        fourierNoiseMagnitudes
      };
    };

    setData(generateData());
  }, []);

  if (!data) {
    return <div>Loading visualization...</div>;
  }

  return (
    <div className="page-container">
      <h1>Orthogonality and Noise in Fourier Space</h1>
      
      <section className="explanation">
        <h2>Orthogonality of Fourier Basis</h2>
        <p>
          The Fourier basis vectors are orthogonal to each other, which means they form an independent set of directions.
          When noise is added to a signal, this noise gets distributed across the frequency components while maintaining
          its statistical properties due to the orthogonality of the Fourier transform.
        </p>
        
        <KatexBlock math="\int_{-\infty}^{\infty} e^{i\omega_1 t} \cdot e^{-i\omega_2 t} dt = 2\pi\delta(\omega_1 - \omega_2)" />
        
        <p>
          This orthogonality property ensures that noise in the time domain translates to noise with similar 
          statistical properties in the frequency domain, just distributed differently.
        </p>
      </section>

      <section className="visualization">
        <h2>Visualization</h2>
        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.timeIndices,
                y: data.signal,
                type: 'scatter',
                mode: 'lines',
                name: 'Clean Signal'
              },
              {
                x: data.timeIndices,
                y: data.noisySignal,
                type: 'scatter',
                mode: 'lines',
                name: 'Noisy Signal',
                line: { color: 'red' }
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Time Domain Signal with Noise',
              xaxis: { title: 'Time' },
              yaxis: { title: 'Amplitude' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.frequencyIndices.slice(0, data.frequencyIndices.length/2),
                y: data.magnitudes.slice(0, data.magnitudes.length/2),
                type: 'scatter',
                mode: 'lines',
                name: 'Clean Signal Spectrum'
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Frequency Domain - Clean Signal',
              xaxis: { title: 'Frequency' },
              yaxis: { title: 'Magnitude' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.frequencyIndices.slice(0, data.frequencyIndices.length/2),
                y: data.noisyMagnitudes.slice(0, data.noisyMagnitudes.length/2),
                type: 'scatter',
                mode: 'lines',
                name: 'Noisy Signal Spectrum',
                line: { color: 'red' }
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Frequency Domain - Noisy Signal',
              xaxis: { title: 'Frequency' },
              yaxis: { title: 'Magnitude' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.frequencyIndices.slice(0, data.frequencyIndices.length/2),
                y: data.fourierNoiseMagnitudes.slice(0, data.fourierNoiseMagnitudes.length/2),
                type: 'scatter',
                mode: 'lines',
                name: 'Noise in Fourier Domain',
                line: { color: 'green' }
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Noise Distribution in Frequency Domain',
              xaxis: { title: 'Frequency' },
              yaxis: { title: 'Magnitude' }
            }}
          />
        </div>
      </section>

      <section className="future-work">
        <h3>Interactive Elements to be Added:</h3>
        <ul>
          <li>Adjustable noise level slider</li>
          <li>Statistical analysis of noise distribution in both domains</li>
          <li>Demonstration of how changing the basis affects noise distribution</li>
          <li>Visualization of the orthogonality property</li>
        </ul>
      </section>
    </div>
  );
};

export default OrthogonalityPage; 