import React from 'react';
import { Card, CardContent, Typography, Box, IconButton, Button } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import { styled } from '@mui/material/styles';

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

const UploadButton = styled(Button)(({ theme }) => ({
  width: '100%',
  height: '100px',
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.background.paper,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: theme.spacing(1),
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
    border: `2px dashed ${theme.palette.primary.dark}`,
  },
}));

const UploadSection = ({ title, onUpload, file, onClear }) => {
  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        <UploadButton
          component="label"
          variant="outlined"
          startIcon={<CloudUploadIcon />}
        >
          {file ? 'Change File' : 'Upload File'}
          <VisuallyHiddenInput 
            type="file" 
            onChange={onUpload} 
            accept="audio/*" 
          />
        </UploadButton>
        {file && (
          <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2" color="text.secondary">
              {file}
            </Typography>
            <IconButton 
              size="small" 
              onClick={onClear}
              color="error"
            >
              <DeleteIcon />
            </IconButton>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default UploadSection; 