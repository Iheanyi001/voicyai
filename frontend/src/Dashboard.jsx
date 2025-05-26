import React, { useState, useRef, useEffect, lazy, Suspense } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Button, 
  Container, 
  Typography, 
  TextField, 
  Paper, 
  Grid, 
  CircularProgress, 
  Alert, 
  Divider,
  Card,
  CardContent,
  Stack,
  IconButton,
  Tooltip,
  useTheme,
  alpha,
  styled,
  Tabs,
  Tab
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import MicIcon from '@mui/icons-material/Mic';
import TextFieldsIcon from '@mui/icons-material/TextFields';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import DownloadIcon from '@mui/icons-material/Download';
import DeleteIcon from '@mui/icons-material/Delete';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import LogoutIcon from '@mui/icons-material/Logout';
import PersonIcon from '@mui/icons-material/Person';
import SchoolIcon from '@mui/icons-material/School';
import SwitchAccessShortcutIcon from '@mui/icons-material/SwitchAccessShortcut';

// Styled components
const StyledPaper = styled(Paper)(({ theme }) => ({
  background: alpha(theme.palette.background.paper, 0.9),
  backdropFilter: 'blur(10px)',
  borderRadius: '24px',
  padding: theme.spacing(4),
  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
}));

const StyledCard = styled(Card)(({ theme }) => ({
  background: alpha(theme.palette.background.paper, 0.8),
  backdropFilter: 'blur(10px)',
  borderRadius: '16px',
  transition: 'all 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 12px 24px rgba(0,0,0,0.1)',
  },
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

// Lazy load components
const AudioPlayer = lazy(() => import('./components/AudioPlayer'));
const UploadSection = lazy(() => import('./components/UploadSection'));
const TranscriptionSection = lazy(() => import('./components/TranscriptionSection'));
const VoiceTraining = lazy(() => import('./components/VoiceTraining'));
const VoiceModelSelector = lazy(() => import('./components/VoiceModelSelector'));

const Dashboard = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [sourceAudio, setSourceAudio] = useState(null);
  const [targetVoice, setTargetVoice] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [inputMode, setInputMode] = useState('audio');
  const [convertedAudioUrl, setConvertedAudioUrl] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const audioRef = useRef(null);
  const [userType, setUserType] = useState('free');
  const [selectedVoiceModel, setSelectedVoiceModel] = useState('');

  // Optimize initial load
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('token');
      if (!token) {
        navigate('/login');
        return;
      }

      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout

        const response = await fetch('/api/verify-token', {
          headers: {
            'Authorization': `Bearer ${token}`
          },
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error('Invalid token');
        }

        const data = await response.json();
        setUserType(data.user?.user_type || 'free');
        setIsAuthenticated(true);
      } catch (err) {
        if (err.name === 'AbortError') {
          setError('Server response timeout. Please try again.');
        } else {
          localStorage.removeItem('token');
          navigate('/login');
        }
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [navigate]);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleSourceAudioUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

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
        throw new Error(errorData.detail || 'Failed to upload source audio');
      }

      const data = await response.json();
      setSourceAudio(data.filename);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleTargetVoiceUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

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
        throw new Error(errorData.detail || 'Failed to upload target voice');
      }

      const data = await response.json();
      setTargetVoice(data.filename);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleTranscribe = async () => {
    if (!sourceAudio) {
      setError('Please upload source audio first');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const formData = new FormData();
      
      // Get the actual file from the uploads directory
      const response = await fetch(`/api/uploads/${sourceAudio}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch audio file');
      }
      
      const audioBlob = await response.blob();
      // Get the file extension from the original filename
      const fileExtension = sourceAudio.split('.').pop().toLowerCase();
      formData.append('audio', audioBlob, `audio.${fileExtension}`);

      const transcribeResponse = await fetch('/api/transcribe', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
      });

      if (!transcribeResponse.ok) {
        const errorData = await transcribeResponse.json();
        throw new Error(errorData.detail || 'Failed to transcribe audio');
      }

      const data = await transcribeResponse.json();
      setTranscript(data.transcription);
      setSuccess('Audio transcribed successfully');
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleVoiceModelSelect = (modelName) => {
    setSelectedVoiceModel(modelName);
    // Set targetVoice to the selected model name
    setTargetVoice(modelName);
  };

  const handleConvert = async () => {
    if (inputMode === 'audio' && !sourceAudio) {
      setError('Please upload source audio first');
      return;
    }

    if (inputMode === 'text' && !transcript) {
      setError('Please enter text to convert');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setConvertedAudioUrl(null);

    try {
      const token = localStorage.getItem('token');
      let body, endpoint;
      if (inputMode === 'audio') {
        if (!transcript) {
          await handleTranscribe();
        }
        
        endpoint = '/api/convert';
        body = JSON.stringify({
          sourceAudio: sourceAudio,
          targetVoice: targetVoice || selectedVoiceModel || null,
          userType: userType,
          transcript: transcript,
        });
      } else {
        endpoint = '/api/text-to-speech';
        body = JSON.stringify({
          targetVoice: targetVoice || selectedVoiceModel || null,
          userType: userType,
          text: transcript,
        });
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to convert voice');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      setConvertedAudioUrl(url);
      
      if (audioRef.current) {
        audioRef.current.src = url;
        audioRef.current.play();
      }

      setSuccess('Voice converted successfully');
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (convertedAudioUrl) {
      const a = document.createElement('a');
      a.href = convertedAudioUrl;
      a.download = 'converted_voice.wav';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const clearFile = (type) => {
    if (type === 'source') {
      setSourceAudio(null);
    } else {
      setTargetVoice(null);
    }
  };

  const renderVoiceSelection = () => {
    return (
      <Box>
        <Typography variant="subtitle1" gutterBottom>
          Voice Model
        </Typography>
        {userType === 'paid' ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Suspense fallback={<CircularProgress size={24} />}>
              <VoiceModelSelector 
                value={selectedVoiceModel} 
                onChange={handleVoiceModelSelect} 
              />
            </Suspense>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <SwitchAccessShortcutIcon fontSize="small" />
              Or upload a custom voice below
            </Typography>
          </Box>
        ) : (
          <Alert severity="info" sx={{ mb: 2 }}>
            Voice models are available for premium users only. <Button size="small" onClick={() => navigate('/package')}>Upgrade</Button>
          </Alert>
        )}
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Upload Custom Voice (Optional)
          </Typography>
          <UploadButton
            component="label"
            variant="outlined"
            startIcon={<VolumeUpIcon sx={{ fontSize: 40 }} />}
          >
            {targetVoice ? 'Change Voice' : 'Upload Voice'}
            <VisuallyHiddenInput type="file" onChange={handleTargetVoiceUpload} accept="audio/*" />
          </UploadButton>
          {targetVoice && (
            <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2" color="text.secondary">
                {targetVoice}
              </Typography>
              <IconButton size="small" onClick={() => clearFile('target')}>
                <DeleteIcon />
              </IconButton>
            </Box>
          )}
        </Box>
      </Box>
    );
  };

  if (isLoading) {
    return (
      <Box sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      }}>
        <CircularProgress size={60} sx={{ color: 'white' }} />
      </Box>
    );
  }

  return (
    <Box sx={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      py: 4,
    }}>
      <Container maxWidth="lg">
        {/* Header */}
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          mb: 4,
        }}>
          <Typography 
            variant="h4" 
            sx={{ 
              fontWeight: 700,
              background: 'linear-gradient(45deg, #fff, #e0e0e0)',
              backgroundClip: 'text',
              textFillColor: 'transparent',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Voice Cloning Dashboard
          </Typography>
          <Stack direction="row" spacing={2}>
            <StyledButton
              variant="outlined"
              startIcon={<PersonIcon />}
              onClick={() => navigate('/package')}
              sx={{
                color: 'white',
                borderColor: 'white',
                '&:hover': {
                  borderColor: 'white',
                  backgroundColor: alpha(theme.palette.common.white, 0.1),
                },
              }}
            >
              {userType === 'paid' ? 'Premium' : 'Free'}
            </StyledButton>
            <StyledButton
              variant="outlined"
              startIcon={<LogoutIcon />}
              onClick={() => {
                localStorage.removeItem('token');
                navigate('/login');
              }}
              sx={{
                color: 'white',
                borderColor: 'white',
                '&:hover': {
                  borderColor: 'white',
                  backgroundColor: alpha(theme.palette.common.white, 0.1),
                },
              }}
            >
              Logout
            </StyledButton>
          </Stack>
        </Box>

        {/* Tab Navigation */}
        <Box sx={{ width: '100%', mb: 3 }}>
          <StyledPaper>
            <Tabs 
              value={tabValue} 
              onChange={handleTabChange} 
              aria-label="dashboard tabs"
              textColor="primary"
              indicatorColor="primary"
              sx={{ 
                borderRadius: '12px',
                '& .MuiTab-root': {
                  minWidth: 120,
                  fontWeight: 500,
                  borderRadius: '12px',
                  transition: 'all 0.2s',
                  '&:hover': {
                    opacity: 0.8,
                  },
                },
              }}
            >
              <Tab icon={<VolumeUpIcon />} label="Voice Conversion" />
              {userType === 'paid' && <Tab icon={<SchoolIcon />} label="Voice Training" />}
            </Tabs>
          </StyledPaper>
        </Box>

        {/* Tab Content */}
        <Suspense fallback={<CircularProgress />}>
          {tabValue === 0 ? (
            <>
              {/* Input Mode Selection */}
              <Box sx={{ mb: 4, display: 'flex', justifyContent: 'center', gap: 2 }}>
                <StyledButton
                  variant={inputMode === 'audio' ? 'contained' : 'outlined'}
                  onClick={() => setInputMode('audio')}
                  startIcon={<MicIcon />}
                  sx={{
                    minWidth: '200px',
                    ...(inputMode !== 'audio' && {
                      color: 'white',
                      borderColor: 'white',
                      '&:hover': {
                        borderColor: 'white',
                        backgroundColor: alpha(theme.palette.common.white, 0.1),
                      },
                    }),
                  }}
                >
                  Audio Input
                </StyledButton>
                <StyledButton
                  variant={inputMode === 'text' ? 'contained' : 'outlined'}
                  onClick={() => setInputMode('text')}
                  startIcon={<TextFieldsIcon />}
                  sx={{
                    minWidth: '200px',
                    ...(inputMode !== 'text' && {
                      color: 'white',
                      borderColor: 'white',
                      '&:hover': {
                        borderColor: 'white',
                        backgroundColor: alpha(theme.palette.common.white, 0.1),
                      },
                    }),
                  }}
                >
                  Text Input
                </StyledButton>
              </Box>

              {/* Main Content */}
              <Grid container spacing={4}>
                {/* Input Section */}
                <Grid item xs={12} md={6}>
                  <StyledPaper>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      Input
                    </Typography>
                    <Stack spacing={3}>
                      {inputMode === 'audio' ? (
                        <>
                          <Box>
                            <Typography variant="subtitle1" gutterBottom>
                              Source Audio
                            </Typography>
                            <UploadButton
                              component="label"
                              variant="outlined"
                              startIcon={<CloudUploadIcon sx={{ fontSize: 40 }} />}
                            >
                              {sourceAudio ? 'Change Audio' : 'Upload Audio'}
                              <VisuallyHiddenInput type="file" onChange={handleSourceAudioUpload} accept="audio/*" />
                            </UploadButton>
                            {sourceAudio && (
                              <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="body2" color="text.secondary">
                                  {sourceAudio}
                                </Typography>
                                <IconButton size="small" onClick={() => clearFile('source')}>
                                  <DeleteIcon />
                                </IconButton>
                              </Box>
                            )}
                          </Box>

                          {renderVoiceSelection()}
                        </>
                      ) : (
                        <>
                          <Box>
                            <Typography variant="subtitle1" gutterBottom>
                              Enter Text
                            </Typography>
                            <TextField
                              fullWidth
                              multiline
                              rows={6}
                              value={transcript}
                              onChange={(e) => setTranscript(e.target.value)}
                              placeholder="Enter the text you want to convert to speech..."
                              variant="outlined"
                              sx={{
                                '& .MuiOutlinedInput-root': {
                                  borderRadius: '12px',
                                  backgroundColor: alpha(theme.palette.background.paper, 0.8),
                                  '&:hover fieldset': {
                                    borderColor: theme.palette.primary.main,
                                  },
                                  '&.Mui-focused fieldset': {
                                    borderWidth: '2px',
                                  },
                                },
                              }}
                            />
                          </Box>

                          {renderVoiceSelection()}
                        </>
                      )}
                    </Stack>
                  </StyledPaper>
                </Grid>

                {/* Output Section */}
                <Grid item xs={12} md={6}>
                  <StyledPaper>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      Output
                    </Typography>
                    <Stack spacing={3}>
                      <Box>
                        <Typography variant="subtitle1" gutterBottom>
                          Converted Audio
                        </Typography>
                        {convertedAudioUrl ? (
                          <Box sx={{ mt: 2 }}>
                            <Suspense fallback={<CircularProgress />}>
                              <AudioPlayer audioUrl={convertedAudioUrl} />
                            </Suspense>
                            <StyledButton
                              fullWidth
                              variant="contained"
                              startIcon={<DownloadIcon />}
                              onClick={handleDownload}
                              sx={{ mt: 2 }}
                            >
                              Download Audio
                            </StyledButton>
                          </Box>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            Convert {inputMode === 'audio' ? 'audio' : 'text'} to see the result here
                          </Typography>
                        )}
                      </Box>
                    </Stack>
                  </StyledPaper>
                </Grid>

                {/* Transcription Section */}
                {inputMode === 'audio' && (
                  <Grid item xs={12}>
                    <StyledPaper>
                      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        Transcription
                      </Typography>
                      <Stack spacing={2}>
                        <TextField
                          fullWidth
                          multiline
                          rows={6}
                          value={transcript}
                          onChange={(e) => setTranscript(e.target.value)}
                          placeholder="Transcribed text will appear here. You can edit it before converting..."
                          variant="outlined"
                          sx={{
                            '& .MuiOutlinedInput-root': {
                              borderRadius: '12px',
                              backgroundColor: alpha(theme.palette.background.paper, 0.8),
                              '&:hover fieldset': {
                                borderColor: theme.palette.primary.main,
                              },
                              '&.Mui-focused fieldset': {
                                borderWidth: '2px',
                              },
                            },
                          }}
                        />
                        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                          <StyledButton
                            variant="outlined"
                            onClick={() => setTranscript('')}
                            disabled={!transcript}
                          >
                            Clear
                          </StyledButton>
                          <StyledButton
                            variant="contained"
                            onClick={handleTranscribe}
                            disabled={!sourceAudio || isProcessing}
                            startIcon={isProcessing ? <CircularProgress size={20} /> : <TextFieldsIcon />}
                          >
                            {isProcessing ? 'Transcribing...' : 'Transcribe'}
                          </StyledButton>
                        </Box>
                      </Stack>
                    </StyledPaper>
                  </Grid>
                )}

                {/* Action Section */}
                <Grid item xs={12}>
                  <StyledPaper>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      Actions
                    </Typography>
                    <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
                      <StyledButton
                        variant="contained"
                        startIcon={<PlayArrowIcon />}
                        onClick={handleConvert}
                        disabled={isProcessing || (inputMode === 'audio' ? !sourceAudio : !transcript)}
                      >
                        {isProcessing ? 'Processing...' : 'Convert'}
                      </StyledButton>
                      {isProcessing && <CircularProgress size={24} />}
                    </Stack>
                  </StyledPaper>
                </Grid>
              </Grid>
            </>
          ) : (
            <VoiceTraining />
          )}
        </Suspense>

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
      </Container>
    </Box>
  );
};

export default Dashboard;
