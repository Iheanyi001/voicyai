import React from "react";
import "./Welcome.css";

const Welcome = () => {
  return (
    <div className="welcome-page">
      {/* Navigation */}
      <nav className="nav-container">
        <div className="container flex justify-between items-center">
          <div className="flex items-center">
            <a href="/" className="text-xl font-bold">VoiceClone AI</a>
          </div>
          <div className="hidden md:flex items-center gap-6">
            <a href="/login" className="btn btn-outline">Sign In</a>
            <a href="/register" className="btn btn-primary">Get Started</a>
          </div>
          <button className="md:hidden">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-16 6h16"/>
            </svg>
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="container mx-auto px-4">
          <h1 className="hero-title">
            AI Voice Cloning<br />
            Made Simple
          </h1>
          <p className="hero-subtitle">
            Create naturally expressive voices for your content using state-of-the-art AI technology
          </p>
          <div className="cta-container">
            <a href="/register" className="btn btn-primary">
              Start for Free
            </a>
            <a href="/login" className="btn btn-outline">
              Sign In
            </a>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="features-grid">
          <div className="feature-card">
            <svg className="feature-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" stroke="currentColor" strokeWidth="2"/>
              <path d="M19 12C19 13.8565 18.2625 15.637 16.9497 16.9497C15.637 18.2625 13.8565 19 12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 19C10.1435 19 8.36301 18.2625 7.05025 16.9497C5.7375 15.637 5 13.8565 5 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 5C13.8565 5 15.637 5.7375 16.9497 7.05025C18.2625 8.36301 19 10.1435 19 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 5C10.1435 5 8.36301 5.7375 7.05025 7.05025C5.7375 8.36301 5 10.1435 5 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <h3 className="feature-title">Real-Time Voice Cloning</h3>
            <p className="feature-description">
              Clone any voice in seconds with just a short audio sample. Perfect for content creators and developers.
            </p>
          </div>

          <div className="feature-card">
            <svg className="feature-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 6V18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M7 9V15" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M17 9V15" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
            </svg>
            <h3 className="feature-title">Natural Expression</h3>
            <p className="feature-description">
              Advanced AI technology ensures natural intonation and emotion in every generated voice.
            </p>
          </div>

          <div className="feature-card">
            <svg className="feature-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M20 7L12 3L4 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M4 7V17L12 21L20 17V7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 12L12 21" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 12L20 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 12L4 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <h3 className="feature-title">Versatile Integration</h3>
            <p className="feature-description">
              Easy to integrate with your existing workflow. Perfect for videos, podcasts, and applications.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="text-sm">© 2025 VoiceClone AI. All rights reserved.</div>
          <div className="footer-links">
            <a href="/privacy" className="footer-link">Privacy Policy</a>
            <a href="/terms" className="footer-link">Terms of Service</a>
            <a href="/contact" className="footer-link">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Welcome;
