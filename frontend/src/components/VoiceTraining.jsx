import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  TextField, 
  CircularProgress, 
  Alert, 
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Chip,
  Paper,
  Grid,
  Stack,
  Divider,
  alpha,
  styled,
  Tooltip
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import AudioFileIcon from '@mui/icons-material/AudioFile';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import MicIcon from '@mui/icons-material/Mic';
import InfoIcon from '@mui/icons-material/Info';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import RefreshIcon from '@mui/icons-material/Refresh';

const StyledPaper = styled(Paper)(({ theme }) => ({
  background: alpha(theme.palette.background.paper, 0.9),
  backdropFilter: 'blur(10px)',
  borderRadius: '24px',
  padding: theme.spacing(4),
  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
}));

const UploadButton = styled(Button)(({ theme }) => ({
  height: '120px',
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: '16px',
  backgroundColor: alpha(theme.palette.primary.main, 0.05),
  transition: 'all 0.3s ease-in-out',
  '&:hover': {
    backgroundColor: alpha(theme.palette.primary.main, 0.1),
    transform: 'translateY(-2px)',
    boxShadow: '0 6px 16px rgba(0,0,0,0.1)',
  },
}));

const StyledButton = styled(Button)(({ theme }) => ({
  borderRadius: '12px',
  padding: '12px 24px',
  textTransform: 'none',
  fontSize: '1rem',
  fontWeight: 600,
  transition: 'all 0.2s ease-in-out',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 6px 16px rgba(0,0,0,0.1)',
  },
}));

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const VoiceTraining = () => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [voiceName, setVoiceName] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [voiceModels, setVoiceModels] = useState([]);
  
  // Fetch existing voice models on component mount
  useEffect(() => {
    fetchVoiceModels();
  }, []);
  
  const fetchVoiceModels = async () => {
    try {
      setError(null);
      const token = localStorage.getItem('token');
      
      // Add a cache-busting query parameter and cache control header
      const response = await fetch(`/api/voice-models?_nocache=${Date.now()}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Cache-Control': 'no-cache'
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch voice models');
      }
      
      const data = await response.json();
      setVoiceModels(data.models || []);
      console.log('Fetched voice models:', data.models);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching voice models:', err);
    }
  };
  
  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    
    setError(null);
    
    // Upload each file
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const formData = new FormData();
      formData.append('audio', file);
      
      try {
        const token = localStorage.getItem('token');
        const response = await fetch('/api/upload', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          },
          body: formData,
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || `Failed to upload file ${file.name}`);
        }
        
        const data = await response.json();
        setUploadedFiles(prev => [...prev, data.filename]);
      } catch (err) {
        setError(err.message);
      }
    }
  };
  
  const handleRemoveFile = (filename) => {
    setUploadedFiles(prev => prev.filter(file => file !== filename));
  };
  
  const handleTrainModel = async () => {
    if (uploadedFiles.length === 0) {
      setError('Please upload at least one audio file');
      return;
    }
    
    if (!voiceName) {
      setError('Please enter a name for your voice model');
      return;
    }
    
    // Validate voice name (alphanumeric and underscores only)
    if (!/^[a-zA-Z0-9_]+$/.test(voiceName)) {
      setError('Voice name can only contain letters, numbers, and underscores');
      return;
    }
    
    setIsTraining(true);
    setError(null);
    setSuccess(null);
    
    try {
      console.log('Starting voice model training for:', voiceName);
      console.log('Using audio files:', uploadedFiles);
      
      const token = localStorage.getItem('token');
      const requestData = {
        audioFiles: uploadedFiles,
        voiceName: voiceName,
        userType: 'paid'
      };
      
      console.log('Sending request data:', requestData);
      
      const response = await fetch('/api/train-voice', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(requestData),
      });
      
      console.log('Training response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to train voice model');
      }
      
      const data = await response.json();
      console.log('Training success data:', data);
      setSuccess(`Voice model "${data.model_name}" trained successfully!`);
      
      // Reset form
      setUploadedFiles([]);
      setVoiceName('');
      
      // Wait a moment and then refresh voice models list
      console.log('Waiting to refresh voice models list...');
      setTimeout(() => {
        console.log('Refreshing voice models list...');
        fetchVoiceModels();
      }, 1000);
    } catch (err) {
      setError(err.message);
      console.error('Error training voice model:', err);
    } finally {
      setIsTraining(false);
    }
  };
  
  return (
    <>
      <StyledPaper sx={{ mb: 4 }}>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
          Train Custom Voice Model
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Upload audio samples of the voice you want to clone. For best results, use clear recordings with minimal background noise, 
          at least 30 seconds to 2 minutes in total length.
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Upload Voice Samples
              </Typography>
              <UploadButton
                component="label"
                variant="outlined"
                startIcon={<CloudUploadIcon sx={{ fontSize: 40 }} />}
                fullWidth
              >
                Upload Audio Samples
                <VisuallyHiddenInput
                  type="file"
                  onChange={handleFileUpload}
                  accept="audio/*"
                  multiple
                />
              </UploadButton>
              
              {uploadedFiles.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    Uploaded Files ({uploadedFiles.length})
                  </Typography>
                  <List>
                    {uploadedFiles.map((file, index) => (
                      <ListItem
                        key={index}
                        secondaryAction={
                          <IconButton edge="end" onClick={() => handleRemoveFile(file)}>
                            <DeleteIcon />
                          </IconButton>
                        }
                      >
                        <ListItemIcon>
                          <AudioFileIcon />
                        </ListItemIcon>
                        <ListItemText 
                          primary={file}
                          primaryTypographyProps={{ noWrap: true }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Voice Model Name
              </Typography>
              <TextField
                fullWidth
                placeholder="Enter a unique name (e.g. my_custom_voice)"
                value={voiceName}
                onChange={(e) => setVoiceName(e.target.value)}
                helperText="Use only letters, numbers, and underscores"
                sx={{ mb: 3 }}
              />
              
              <StyledButton
                variant="contained"
                startIcon={isTraining ? <CircularProgress size={20} /> : <MicIcon />}
                onClick={handleTrainModel}
                disabled={isTraining || uploadedFiles.length === 0 || !voiceName}
                fullWidth
                sx={{ mt: 2 }}
              >
                {isTraining ? 'Training...' : 'Train Voice Model'}
              </StyledButton>
            </Box>
          </Grid>
        </Grid>
      </StyledPaper>
      
      {/* Existing Voice Models */}
      <StyledPaper>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" gutterBottom={false} sx={{ fontWeight: 600 }}>
            Your Voice Models
          </Typography>
          <IconButton 
            color="primary" 
            onClick={fetchVoiceModels} 
            disabled={isTraining}
            size="small"
            sx={{ p: 1 }}
          >
            <Tooltip title="Refresh voice models">
              <RefreshIcon />
            </Tooltip>
          </IconButton>
        </Box>
        
        {voiceModels.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            You haven't created any voice models yet.
          </Typography>
        ) : (
          <List>
            {voiceModels.map((model, index) => (
              <ListItem key={index}>
                <ListItemIcon>
                  <VolumeUpIcon />
                </ListItemIcon>
                <ListItemText 
                  primary={model.name}
                  secondary={`Created: ${new Date(model.created * 1000).toLocaleString()}`}
                />
                <Chip 
                  icon={<CheckCircleIcon />} 
                  label="Ready to use" 
                  color="success" 
                  size="small"
                  sx={{ ml: 2 }}
                />
              </ListItem>
            ))}
          </List>
        )}
        
        <Box sx={{ mt: 2, display: 'flex', alignItems: 'center' }}>
          <InfoIcon fontSize="small" color="info" sx={{ mr: 1 }} />
          <Typography variant="body2" color="info.main">
            Use these models as "Target Voice" in the voice conversion section.
          </Typography>
        </Box>
      </StyledPaper>
      
      {/* Messages */}
      {(error || success) && (
        <Box sx={{ mt: 4 }}>
          {error && (
            <Alert severity="error" sx={{ borderRadius: '12px' }}>
              {error}
            </Alert>
          )}
          {success && (
            <Alert severity="success" sx={{ borderRadius: '12px' }}>
              {success}
            </Alert>
          )}
        </Box>
      )}
    </>
  );
};

export default VoiceTraining; 