import React from 'react';
import { Card, CardContent, Typography, TextField, Button } from '@mui/material';
import MicIcon from '@mui/icons-material/Mic';

const TranscriptionSection = ({ mode, value, onChange, onTranscribe, disabled }) => {
  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {mode === 'audio' ? 'Transcription' : 'Text Input'}
        </Typography>
        <TextField
          fullWidth
          multiline
          rows={4}
          value={value}
          onChange={onChange}
          placeholder={mode === 'audio' ? 'Transcription will appear here...' : 'Enter text to convert...'}
          variant="outlined"
          sx={{ mb: 2 }}
        />
        {mode === 'audio' && (
          <Button
            variant="contained"
            onClick={onTranscribe}
            disabled={disabled}
            startIcon={<MicIcon />}
          >
            Transcribe
          </Button>
        )}
      </CardContent>
    </Card>
  );
};

export default TranscriptionSection; 