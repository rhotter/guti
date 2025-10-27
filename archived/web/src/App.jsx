import React, { useEffect, useState } from 'react';
import './App.css';

// Import visualization pages
import ChannelCodingPage from './pages/ChannelCodingPage';
import EigenvaluesPage from './pages/EigenvaluesPage';
import FourierPage from './pages/FourierPage';
import LinearModelPage from './pages/LinearModelPage';
import OrthogonalityPage from './pages/OrthogonalityPage';

function App() {
  const [showBackToTop, setShowBackToTop] = useState(false);

  // Handle scroll events to show/hide the back to top button
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 300) {
        setShowBackToTop(true);
      } else {
        setShowBackToTop(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  // Function to handle scrolling back to top
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  // Function to handle active navigation highlighting
  useEffect(() => {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.main-nav a');

    const handleNavHighlight = () => {
      let scrollY = window.scrollY;
      
      sections.forEach(section => {
        const sectionHeight = section.offsetHeight;
        const sectionTop = section.offsetTop - 100;
        const sectionId = section.getAttribute('id');
        
        if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
          navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${sectionId}`) {
              link.classList.add('active');
            }
          });
        }
      });
    };

    window.addEventListener('scroll', handleNavHighlight);
    return () => {
      window.removeEventListener('scroll', handleNavHighlight);
    };
  }, []);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Interactive Signal Processing Visualizations</h1>
        <nav className="main-nav">
          <ul>
            <li>
              <a href="#intro">Home</a>
            </li>
            <li>
              <a href="#linear-model">Linear Model</a>
            </li>
            <li>
              <a href="#fourier">Fourier Transform</a>
            </li>
            <li>
              <a href="#eigenvalues">Eigenvalues</a>
            </li>
            <li>
              <a href="#orthogonality">Orthogonality & Noise</a>
            </li>
            <li>
              <a href="#channel-coding">SNR & Channel Coding</a>
            </li>
          </ul>
        </nav>
      </header>

      <main className="content-container">
        {/* Introduction Section */}
        <section id="intro" className="intro-section">
          <h1>Welcome to Interactive Signal Processing Visualizations</h1>
          
          <p className="intro">
            This application provides interactive visualizations to help understand key 
            concepts in signal processing, linear algebra, and information theory.
          </p>
          
          <div className="visualization-sections">
            <h2>Available Visualizations</h2>
            
            <p>
              Scroll down to explore each visualization, or use the navigation links above to jump to a specific section.
              Each visualization demonstrates a fundamental concept in signal processing and how these concepts connect to one another.
            </p>
          </div>
        </section>
        
        {/* Linear Model Section */}
        <section id="linear-model" className="visualization-section">
          <h2 className="section-header">Linear Model Visualization</h2>
          <div className="section-content">
            <LinearModelPage />
          </div>
        </section>
        
        {/* Fourier Transform Section */}
        <section id="fourier" className="visualization-section">
          <h2 className="section-header">Fourier Transform Visualization</h2>
          <div className="section-content">
            <FourierPage />
          </div>
        </section>
        
        {/* Eigenvalues Section */}
        <section id="eigenvalues" className="visualization-section">
          <h2 className="section-header">Eigenvalues Visualization</h2>
          <div className="section-content">
            <EigenvaluesPage />
          </div>
        </section>
        
        {/* Orthogonality Section */}
        <section id="orthogonality" className="visualization-section">
          <h2 className="section-header">Orthogonality and Noise Visualization</h2>
          <div className="section-content">
            <OrthogonalityPage />
          </div>
        </section>
        
        {/* Channel Coding Section */}
        <section id="channel-coding" className="visualization-section">
          <h2 className="section-header">SNR and Channel Coding Visualization</h2>
          <div className="section-content">
            <ChannelCodingPage />
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>
          Interactive Visualizations for Signal Processing and Linear Algebra Concepts
        </p>
      </footer>

      {/* Back to Top Button */}
      {showBackToTop && (
        <div className="back-to-top" onClick={scrollToTop} aria-label="Back to top">
          ↑
        </div>
      )}
    </div>
  );
}

export default App;
