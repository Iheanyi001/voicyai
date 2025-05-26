import React, { useEffect, useRef } from 'react';
import { Card, CardContent, Typography, Box, Button } from '@mui/material';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import DownloadIcon from '@mui/icons-material/Download';

const AudioPlayer = ({ audioUrl, onDownload, audioRef }) => {
  const audioElementRef = useRef(null);

  useEffect(() => {
    if (audioUrl && audioElementRef.current) {
      audioElementRef.current.src = audioUrl;
    }
  }, [audioUrl]);

  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Converted Audio
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <VolumeUpIcon color="primary" />
            <audio 
              ref={(el) => {
                audioElementRef.current = el;
                if (audioRef) audioRef.current = el;
              }}
              controls 
              style={{ width: '100%' }}
              controlsList="nodownload"
            >
              <source src={audioUrl} type="audio/wav" />
              Your browser does not support the audio element.
            </audio>
          </Box>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={onDownload}
            fullWidth
          >
            Download Audio
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default AudioPlayer; 