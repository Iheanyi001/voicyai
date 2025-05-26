import React, { useState, useEffect } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  CircularProgress,
  Alert,
  Stack,
  Chip,
  alpha,
  styled,
  IconButton,
  Tooltip
} from '@mui/material';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import RefreshIcon from '@mui/icons-material/Refresh';

const StyledSelect = styled(Select)(({ theme }) => ({
  borderRadius: '12px',
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: alpha(theme.palette.primary.main, 0.4),
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: theme.palette.primary.main,
  },
  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
    borderColor: theme.palette.primary.main,
    borderWidth: '2px',
  }
}));

const VoiceModelSelector = ({ onChange, value }) => {
  const [voiceModels, setVoiceModels] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(Date.now());

  useEffect(() => {
    fetchVoiceModels();
  }, [lastRefresh]); // Re-fetch when lastRefresh changes

  const fetchVoiceModels = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      console.log('Fetching voice models...');
      const token = localStorage.getItem('token');
      console.log('Token exists:', !!token);
      
      // Add a cache-busting parameter to prevent browser caching
      const response = await fetch(`/api/voice-models?_nocache=${Date.now()}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Cache-Control': 'no-cache'
        }
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        throw new Error('Failed to fetch voice models');
      }
      
      const data = await response.json();
      console.log('Voice models data:', data);
      setVoiceModels(data.models || []);
    } catch (err) {
      console.error('Error fetching voice models:', err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRefresh = () => {
    setLastRefresh(Date.now());
  };

  if (error) {
    return (
      <Box>
        <Alert 
          severity="error" 
          sx={{ borderRadius: '12px', mb: 2 }}
          action={
            <IconButton
              aria-label="refresh"
              color="inherit"
              size="small"
              onClick={handleRefresh}
            >
              <RefreshIcon fontSize="inherit" />
            </IconButton>
          }
        >
          Error loading voice models: {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <FormControl fullWidth>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <InputLabel id="voice-model-select-label">Select Voice Model</InputLabel>
          <Box sx={{ ml: 'auto' }}>
            <Tooltip title="Refresh voice models">
              <IconButton 
                onClick={handleRefresh} 
                size="small" 
                color="primary"
                disabled={isLoading}
              >
                {isLoading ? <CircularProgress size={18} /> : <RefreshIcon />}
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        
        <StyledSelect
          labelId="voice-model-select-label"
          id="voice-model-select"
          value={value || ''}
          label="Select Voice Model"
          onChange={(e) => onChange(e.target.value)}
          renderValue={(selected) => (
            <Stack direction="row" alignItems="center" spacing={1}>
              <VolumeUpIcon fontSize="small" />
              <Typography variant="body2">{selected}</Typography>
            </Stack>
          )}
          disabled={isLoading}
        >
          <MenuItem value="">
            <em>None (Use Default Voice)</em>
          </MenuItem>
          {voiceModels.map((model) => (
            <MenuItem key={model.name} value={model.name}>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ width: '100%' }}>
                <VolumeUpIcon fontSize="small" />
                <Typography variant="body2">{model.name}</Typography>
                <Box sx={{ flexGrow: 1 }} />
                <Chip 
                  size="small"
                  label={new Date(model.created * 1000).toLocaleDateString()}
                  color="primary"
                  variant="outlined"
                />
              </Stack>
            </MenuItem>
          ))}
        </StyledSelect>
      </FormControl>
      
      {isLoading && (
        <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
          <CircularProgress size={16} />
          <Typography variant="caption" sx={{ ml: 1 }}>
            Loading voice models...
          </Typography>
        </Box>
      )}
      
      {!isLoading && voiceModels.length === 0 && (
        <Box sx={{ p: 2, mt: 1, borderRadius: '12px', border: '1px dashed', borderColor: 'divider' }}>
          <Typography variant="body2" color="text.secondary" align="center">
            No trained voice models found. Create a model in the Voice Training tab.
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default VoiceModelSelector; 