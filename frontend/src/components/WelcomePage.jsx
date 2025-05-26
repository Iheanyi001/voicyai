import React from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Button, 
  Grid, 
  Card, 
  CardContent,
  useTheme,
  alpha,
  styled,
  useMediaQuery
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import MicIcon from '@mui/icons-material/Mic';
import TextFieldsIcon from '@mui/icons-material/TextFields';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import RecordVoiceOverIcon from '@mui/icons-material/RecordVoiceOver';
import SecurityIcon from '@mui/icons-material/Security';
import SpeedIcon from '@mui/icons-material/Speed';
import { motion } from 'framer-motion';

const MotionBox = motion(Box);
const MotionCard = motion(Card);

const GradientText = styled(Typography)(({ theme }) => ({
  background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
  backgroundClip: 'text',
  textFillColor: 'transparent',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  fontWeight: 700,
}));

const FeatureCard = styled(Card)(({ theme }) => ({
  background: alpha(theme.palette.background.paper, 0.8),
  backdropFilter: 'blur(10px)',
  borderRadius: '24px',
  padding: theme.spacing(3),
  height: '100%',
  transition: 'all 0.3s ease-in-out',
  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
  '&:hover': {
    transform: 'translateY(-8px)',
    boxShadow: '0 12px 24px rgba(0,0,0,0.1)',
    border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
  },
}));

const StyledButton = styled(Button)(({ theme }) => ({
  borderRadius: '12px',
  padding: '12px 32px',
  textTransform: 'none',
  fontSize: '1.1rem',
  fontWeight: 600,
  transition: 'all 0.2s ease-in-out',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 6px 16px rgba(0,0,0,0.1)',
  },
}));

const StatsCard = styled(Card)(({ theme }) => ({
  background: alpha(theme.palette.background.paper, 0.9),
  backdropFilter: 'blur(10px)',
  borderRadius: '16px',
  padding: theme.spacing(2),
  textAlign: 'center',
  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
}));

const WelcomePage = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const features = [
    {
      icon: <MicIcon sx={{ fontSize: 40, color: theme.palette.primary.main }} />,
      title: "Voice Cloning",
      description: "Transform your voice into any style you want with our advanced AI technology."
    },
    {
      icon: <TextFieldsIcon sx={{ fontSize: 40, color: theme.palette.primary.main }} />,
      title: "Text to Speech",
      description: "Convert your text into natural-sounding speech with customizable voices."
    },
    {
      icon: <AutoAwesomeIcon sx={{ fontSize: 40, color: theme.palette.primary.main }} />,
      title: "High Quality",
      description: "Experience crystal clear audio with our state-of-the-art voice synthesis."
    },
    {
      icon: <RecordVoiceOverIcon sx={{ fontSize: 40, color: theme.palette.primary.main }} />,
      title: "Multiple Voices",
      description: "Choose from a variety of voices or create your own custom voice."
    },
    {
      icon: <SecurityIcon sx={{ fontSize: 40, color: theme.palette.primary.main }} />,
      title: "Secure & Private",
      description: "Your voice data is encrypted and protected with enterprise-grade security."
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 40, color: theme.palette.primary.main }} />,
      title: "Fast Processing",
      description: "Get your converted audio in seconds with our optimized processing pipeline."
    }
  ];

  const stats = [
    { value: "10K+", label: "Active Users" },
    { value: "1M+", label: "Voices Generated" },
    { value: "99%", label: "Satisfaction Rate" }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        py: 8,
        display: 'flex',
        alignItems: 'center',
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 0%, transparent 50%)',
          pointerEvents: 'none',
        }
      }}
    >
      <Container maxWidth="lg">
        <MotionBox
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Hero Section */}
          <Grid item xs={12} textAlign="center" mb={6}>
            <MotionBox variants={itemVariants}>
              <GradientText variant={isMobile ? "h3" : "h2"} gutterBottom>
                Welcome to Voice Magic
              </GradientText>
              <Typography 
                variant={isMobile ? "h6" : "h5"} 
                color="white" 
                sx={{ 
                  mb: 4,
                  maxWidth: '800px',
                  mx: 'auto',
                  opacity: 0.9
                }}
              >
                Transform your voice with cutting-edge AI technology. Create, customize, and convert voices with ease.
              </Typography>
              <StyledButton
                variant="contained"
                size="large"
                onClick={() => navigate('/register')}
                sx={{
                  background: 'white',
                  color: theme.palette.primary.main,
                  '&:hover': {
                    background: alpha('#fff', 0.9),
                  },
                }}
              >
                Get Started
              </StyledButton>
            </MotionBox>
          </Grid>

          {/* Stats Section */}
          <Grid container spacing={3} sx={{ mb: 6 }}>
            {stats.map((stat, index) => (
              <Grid item xs={12} sm={4} key={index}>
                <MotionBox variants={itemVariants}>
                  <StatsCard>
                    <Typography variant="h4" color="primary" fontWeight="bold">
                      {stat.value}
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                      {stat.label}
                    </Typography>
                  </StatsCard>
                </MotionBox>
              </Grid>
            ))}
          </Grid>

          {/* Features Section */}
          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <MotionBox variants={itemVariants}>
                  <FeatureCard>
                    <CardContent>
                      <Box sx={{ mb: 2 }}>
                        {feature.icon}
                      </Box>
                      <Typography variant="h6" gutterBottom fontWeight="600">
                        {feature.title}
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        {feature.description}
                      </Typography>
                    </CardContent>
                  </FeatureCard>
                </MotionBox>
              </Grid>
            ))}
          </Grid>

          {/* Call to Action */}
          <Grid item xs={12} textAlign="center" mt={8}>
            <MotionBox variants={itemVariants}>
              <Typography variant="h4" color="white" gutterBottom>
                Ready to Transform Your Voice?
              </Typography>
              <Typography variant="body1" color="white" sx={{ mb: 4, opacity: 0.9 }}>
                Join thousands of users who are already creating amazing voice content.
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
                <StyledButton
                  variant="contained"
                  size="large"
                  onClick={() => navigate('/register')}
                  sx={{
                    background: 'white',
                    color: theme.palette.primary.main,
                    '&:hover': {
                      background: alpha('#fff', 0.9),
                    },
                  }}
                >
                  Start Creating Now
                </StyledButton>
                <StyledButton
                  variant="outlined"
                  size="large"
                  onClick={() => navigate('/login')}
                  sx={{
                    borderColor: 'white',
                    color: 'white',
                    '&:hover': {
                      borderColor: alpha('#fff', 0.9),
                      background: alpha('#fff', 0.1),
                    },
                  }}
                >
                  Sign In
                </StyledButton>
              </Box>
            </MotionBox>
          </Grid>
        </MotionBox>
      </Container>
    </Box>
  );
};

export default WelcomePage; 