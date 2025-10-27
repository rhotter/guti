import { math } from 'mathjs';
import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { KatexBlock } from '../components/KatexBlock';

const LinearModelPage = () => {
  const [data, setData] = useState([]);
  
  useEffect(() => {
    // Example data preparation
    const generateData = () => {
      // Simple vector to be smoothed
      const vectorSize = 50;
      const originalVector = Array(vectorSize).fill(0).map((_, i) => 
        i > 15 && i < 35 ? 1 : 0 + 0.2 * Math.random() - 0.1);
      
      // Generate a simple smoothing matrix (Gaussian kernel)
      const smoothingMatrix = [];
      for (let i = 0; i < vectorSize; i++) {
        const row = [];
        for (let j = 0; j < vectorSize; j++) {
          // Gaussian kernel
          row.push(Math.exp(-0.1 * Math.pow(i - j, 2)));
        }
        smoothingMatrix.push(row);
      }

      // Normalize rows
      for (let i = 0; i < vectorSize; i++) {
        const sum = smoothingMatrix[i].reduce((a, b) => a + b, 0);
        for (let j = 0; j < vectorSize; j++) {
          smoothingMatrix[i][j] /= sum;
        }
      }

      // Apply the smoothing matrix
      const smoothedVector = math.multiply(smoothingMatrix, originalVector);
      
      return {
        originalVector,
        smoothedVector: Array.isArray(smoothedVector) ? smoothedVector : smoothedVector.toArray(),
        smoothingMatrix
      };
    };

    setData(generateData());
  }, []);

  // Render loading state if data is not yet available
  if (data.length === 0) {
    return <div>Loading visualization...</div>;
  }

  return (
    <div className="page-container">
      <h1>Linear Model Visualization</h1>
      
      <section className="explanation">
        <h2>Linear Models as Matrix Operations</h2>
        <p>
          Linear models can be visualized as matrices acting on vectors. 
          Here, we demonstrate how a smoothing operation can be represented 
          as a matrix of Gaussian weights multiplying a vector.
        </p>
        
        <KatexBlock math="y = Ax" />
        
        <p>
          Where A is our matrix of Gaussian weights and x is our input vector.
          The result y is the smoothed version of x.
        </p>
      </section>

      <section className="visualization">
        <h2>Visualization</h2>
        <div className="plot-container">
          <Plot
            data={[
              {
                x: Array.from({ length: data.originalVector.length }, (_, i) => i),
                y: data.originalVector,
                type: 'scatter',
                mode: 'lines',
                name: 'Original Vector'
              },
              {
                x: Array.from({ length: data.smoothedVector.length }, (_, i) => i),
                y: data.smoothedVector,
                type: 'scatter',
                mode: 'lines',
                name: 'Smoothed Vector'
              }
            ]}
            layout={{
              width: 800,
              height: 400,
              title: 'Original vs Smoothed Vector',
              xaxis: { title: 'Position' },
              yaxis: { title: 'Value' }
            }}
          />
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                z: data.smoothingMatrix,
                type: 'heatmap',
                colorscale: 'Viridis'
              }
            ]}
            layout={{
              width: 600,
              height: 500,
              title: 'Smoothing Matrix (Gaussian Kernel)',
              xaxis: { title: 'Column Index' },
              yaxis: { title: 'Row Index' }
            }}
          />
        </div>
      </section>

      <section className="future-work">
        <h3>Interactive Elements to be Added:</h3>
        <ul>
          <li>Adjustable kernel width for the Gaussian smoothing</li>
          <li>Ability to upload custom vectors</li>
          <li>Animated visualization of the smoothing process</li>
        </ul>
      </section>
    </div>
  );
};

export default LinearModelPage; 