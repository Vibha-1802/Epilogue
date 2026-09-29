import React, { useState, useRef } from 'react';
import { UploadCloud, Zap, Loader2 } from 'lucide-react';
import './index.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [annotatedUrl, setAnnotatedUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnnotatedUrl(null); 
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      if (data.annotated_image) {
        setAnnotatedUrl(`data:image/jpeg;base64,${data.annotated_image}`);
      }
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert("Failed to connect to backend server.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>AI Traffic Vision</h1>
        <p>Advanced YOLOv11 Sensor Fusion & Trajectory Analysis</p>
      </div>

      {!previewUrl && (
        <div 
          className="upload-zone" 
          onClick={() => fileInputRef.current.click()}
        >
          <UploadCloud className="upload-icon" />
          <h2>Drop a traffic photo here</h2>
          <p style={{color: '#64748b'}}>or click to browse from your computer</p>
          <input 
            type="file" 
            style={{ display: 'none' }} 
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            id="file-upload"
          />
        </div>
      )}

      {previewUrl && (
        <>
          <div className="preview-container">
            <div className="image-card">
              <h3>Original Input</h3>
              <img src={previewUrl} alt="Original" />
              {!annotatedUrl && (
                <div style={{textAlign: 'center', marginTop: '1rem'}}>
                    <button id="analyze-button" className="analyze-btn" onClick={handleAnalyze} disabled={isAnalyzing}>
                    {isAnalyzing ? <Loader2 className="spinner" /> : <Zap />}
                    {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
                    </button>
                    <br />
                    <button style={{marginTop: '1rem', background: 'transparent', color: '#94a3b8', border: 'none', cursor: 'pointer', textDecoration: 'underline'}} onClick={() => {
                        setPreviewUrl(null);
                        setAnnotatedUrl(null);
                    }}>Choose another image</button>
                </div>
              )}
            </div>

            {annotatedUrl && (
              <div className="image-card">
                <h3>YOLO Analysis Output</h3>
                <img id="annotated-image" src={annotatedUrl} alt="Annotated" />
              </div>
            )}
          </div>
          
          {annotatedUrl && (
            <button style={{marginTop: '2rem', background: 'transparent', color: '#94a3b8', border: 'none', cursor: 'pointer', textDecoration: 'underline'}} onClick={() => {
                setPreviewUrl(null);
                setAnnotatedUrl(null);
            }}>Analyze Another Photo</button>
          )}
        </>
      )}
    </div>
  );
}

export default App;
