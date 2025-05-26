import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { 
  Box, 
  Button, 
  Typography, 
  Grid, 
  Card, 
  CardContent, 
  CardActions, 
  Chip,
  Container,
  useTheme,
  alpha,
  styled,
  Stack,
  Divider,
  CircularProgress,
  Alert,
  Snackbar
} from '@mui/material';
import WorkspacePremiumIcon from '@mui/icons-material/WorkspacePremium';
import StarBorderIcon from '@mui/icons-material/StarBorder';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import SecurityIcon from '@mui/icons-material/Security';
import SpeedIcon from '@mui/icons-material/Speed';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import { motion } from 'framer-motion';

const MotionCard = motion(Card);

const StyledCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  borderRadius: '24px',
  transition: 'all 0.3s ease-in-out',
  background: alpha(theme.palette.background.paper, 0.9),
  backdropFilter: 'blur(10px)',
  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
  '&:hover': {
    transform: 'translateY(-8px)',
    boxShadow: '0 12px 24px rgba(0,0,0,0.1)',
    border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
  },
}));

const FeatureItem = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginBottom: theme.spacing(1.5),
  '& svg': {
    color: theme.palette.primary.main,
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

const PriceTag = styled(Typography)(({ theme }) => ({
  fontSize: '3rem',
  fontWeight: 700,
  color: theme.palette.primary.main,
  marginBottom: theme.spacing(1),
}));

const Payment = () => {
  const [status, setStatus] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const navigate = useNavigate();
  const theme = useTheme();

  const features = [
    {
      icon: <VolumeUpIcon />,
      text: "Unlimited voice conversions"
    },
    {
      icon: <SpeedIcon />,
      text: "Priority processing"
    },
    {
      icon: <SecurityIcon />,
      text: "Advanced security features"
    },
    {
      icon: <StarBorderIcon />,
      text: "Premium voice models"
    },
    {
      icon: <CheckCircleOutlineIcon />,
      text: "24/7 support"
    }
  ];

  const handlePayment = async () => {
    setIsProcessing(true);
    setStatus("Processing payment...");
    
    try {
      const token = localStorage.getItem("token");
      const response = await fetch("/api/user/upgrade", {
        method: "POST",
        headers: { 
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
      });

      if (!response.ok) {
        throw new Error('Payment failed');
      }

      localStorage.setItem("user_type", "paid");
      setStatus("Payment successful!");
      setShowSuccess(true);
      
      setTimeout(() => {
        navigate("/dashboard");
      }, 2000);
    } catch (error) {
      setStatus("Payment failed. Please try again.");
    } finally {
      setIsProcessing(false);
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
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          {/* Header */}
          <Grid item xs={12} textAlign="center" mb={4}>
            <Typography 
              variant="h3" 
              color="white" 
              gutterBottom
              sx={{ fontWeight: 700 }}
            >
              Upgrade to Premium
            </Typography>
            <Typography 
              variant="h6" 
              color="white" 
              sx={{ opacity: 0.9, maxWidth: '600px', mx: 'auto' }}
            >
              Unlock all premium features and take your voice cloning to the next level
            </Typography>
          </Grid>

          {/* Pricing Card */}
          <Grid item xs={12} md={6} lg={5} mx="auto">
            <MotionCard
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ textAlign: 'center', mb: 4 }}>
                  <WorkspacePremiumIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h5" gutterBottom fontWeight="600">
                    Premium Plan
                  </Typography>
                  <PriceTag>
                    $50
                  </PriceTag>
                  <Typography variant="body1" color="text.secondary">
                    One-time payment
                  </Typography>
                </Box>

                <Divider sx={{ my: 3 }} />

                <Stack spacing={2}>
                  {features.map((feature, index) => (
                    <FeatureItem key={index}>
                      {feature.icon}
                      <Typography variant="body1">
                        {feature.text}
                      </Typography>
                    </FeatureItem>
                  ))}
                </Stack>
              </CardContent>

              <CardActions sx={{ p: 4, pt: 0 }}>
                <StyledButton
                  fullWidth
                  variant="contained"
                  size="large"
                  onClick={handlePayment}
                  disabled={isProcessing}
                  startIcon={isProcessing ? <CircularProgress size={20} /> : null}
                >
                  {isProcessing ? "Processing..." : "Upgrade Now"}
                </StyledButton>
              </CardActions>
            </MotionCard>
          </Grid>

          {/* Additional Info */}
          <Grid item xs={12} textAlign="center" mt={4}>
            <Typography variant="body2" color="white" sx={{ opacity: 0.8 }}>
              Secure payment processing • 30-day money-back guarantee • Instant access
            </Typography>
          </Grid>
        </Grid>
      </Container>

      <Snackbar
        open={showSuccess}
        autoHideDuration={3000}
        onClose={() => setShowSuccess(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          severity="success" 
          sx={{ 
            width: '100%',
            borderRadius: '12px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
          }}
        >
          Payment successful! Redirecting to dashboard...
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Payment;
