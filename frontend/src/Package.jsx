import React, { useEffect } from "react";
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
  styled
} from '@mui/material';
import WorkspacePremiumIcon from '@mui/icons-material/WorkspacePremium';
import StarBorderIcon from '@mui/icons-material/StarBorder';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';

// Styled components
const StyledCard = styled(Card)(({ theme, isactive }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  borderRadius: '24px',
  transition: 'all 0.3s ease-in-out',
  border: isactive === 'true' ? `2px solid ${theme.palette.primary.main}` : '2px solid transparent',
  background: alpha(theme.palette.background.paper, 0.9),
  backdropFilter: 'blur(10px)',
  '&:hover': {
    transform: 'translateY(-8px)',
    boxShadow: '0 12px 24px rgba(0,0,0,0.1)',
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

const Package = () => {
  const userType = localStorage.getItem("user_type") || "free";
  const navigate = useNavigate();
  const theme = useTheme();

  useEffect(() => {
    // Visiting the package page means first login is complete
    if (localStorage.getItem("first_login") === "true") {
      localStorage.setItem("first_login", "false");
    }
  }, []);

  const features = {
    free: [
      "2 minutes max audio length",
      "Male/Female preset voices",
      "Basic voice cloning",
      "Standard processing speed",
    ],
    premium: [
      "10 minutes max audio length",
      "Upload & manage custom voices",
      "Advanced voice cloning",
      "Priority processing",
      "Unlimited voice models",
      "Premium support",
    ],
  };

  return (
    <Box sx={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      py: 8,
    }}>
      <Container maxWidth="lg">
        <Box sx={{ textAlign: 'center', mb: 8 }}>
          <Typography 
            variant="h3" 
            sx={{ 
              fontWeight: 700,
              mb: 2,
              background: 'linear-gradient(45deg, #fff, #e0e0e0)',
              backgroundClip: 'text',
              textFillColor: 'transparent',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Choose Your Plan
          </Typography>
          <Typography 
            variant="h6" 
            sx={{ 
              color: alpha(theme.palette.common.white, 0.8),
              maxWidth: '600px',
              mx: 'auto',
            }}
          >
            Select the perfect plan for your voice cloning needs
          </Typography>
          <Chip
            icon={userType === "paid" ? <WorkspacePremiumIcon /> : <StarBorderIcon />}
            label={`Current Plan: ${userType === "paid" ? "Premium" : "Free"}`}
            color={userType === "paid" ? "primary" : "default"}
            sx={{ 
              mt: 2,
              px: 2,
              py: 3,
              fontSize: '1rem',
              '& .MuiChip-icon': { fontSize: '1.2rem' },
            }}
          />
        </Box>

        <Grid container spacing={4} justifyContent="center">
          <Grid item xs={12} md={6} lg={5}>
            <StyledCard isactive={userType === "free" ? "true" : "false"}>
              <CardContent sx={{ p: 4, flexGrow: 1 }}>
                <Typography variant="h4" gutterBottom sx={{ fontWeight: 700 }}>
                  Free Plan
                </Typography>
                <Typography variant="h3" sx={{ mb: 3, fontWeight: 700 }}>
                  $0
                  <Typography component="span" variant="subtitle1" sx={{ ml: 1, color: 'text.secondary' }}>
                    /forever
                  </Typography>
                </Typography>
                <Box sx={{ mt: 4 }}>
                  {features.free.map((feature, index) => (
                    <FeatureItem key={index}>
                      <CheckCircleOutlineIcon />
                      <Typography variant="body1">{feature}</Typography>
                    </FeatureItem>
                  ))}
                </Box>
              </CardContent>
              <CardActions sx={{ p: 4, pt: 0 }}>
                <StyledButton
                  fullWidth
                  variant="contained"
                  onClick={() => navigate("/dashboard")}
                  endIcon={<ArrowForwardIcon />}
                >
                  Continue as Free
                </StyledButton>
              </CardActions>
            </StyledCard>
          </Grid>

          <Grid item xs={12} md={6} lg={5}>
            <StyledCard isactive={userType === "paid" ? "true" : "false"}>
              <CardContent sx={{ p: 4, flexGrow: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Typography variant="h4" sx={{ fontWeight: 700 }}>
                    Premium Plan
                  </Typography>
                  <Chip 
                    label="Popular" 
                    color="primary" 
                    size="small"
                    sx={{ 
                      height: '24px',
                      '& .MuiChip-label': { px: 1, fontSize: '0.75rem' },
                    }}
                  />
                </Box>
                <Typography variant="h3" sx={{ mb: 3, fontWeight: 700 }}>
                  $50
                  <Typography component="span" variant="subtitle1" sx={{ ml: 1, color: 'text.secondary' }}>
                    /one-time
                  </Typography>
                </Typography>
                <Box sx={{ mt: 4 }}>
                  {features.premium.map((feature, index) => (
                    <FeatureItem key={index}>
                      <CheckCircleOutlineIcon />
                      <Typography variant="body1">{feature}</Typography>
                    </FeatureItem>
                  ))}
                </Box>
              </CardContent>
              <CardActions sx={{ p: 4, pt: 0 }}>
                {userType === "free" ? (
                  <StyledButton
                    fullWidth
                    variant="contained"
                    color="primary"
                    onClick={() => navigate("/payment")}
                    endIcon={<ArrowForwardIcon />}
                  >
                    Upgrade to Premium
                  </StyledButton>
                ) : (
                  <StyledButton
                    fullWidth
                    variant="contained"
                    onClick={() => navigate("/dashboard")}
                    endIcon={<ArrowForwardIcon />}
                  >
                    Go to Dashboard
                  </StyledButton>
                )}
              </CardActions>
            </StyledCard>
          </Grid>
        </Grid>

        <Box sx={{ textAlign: 'center', mt: 6 }}>
          <StyledButton
            variant="outlined"
            onClick={() => navigate("/dashboard")}
            sx={{
              color: 'white',
              borderColor: 'white',
              '&:hover': {
                borderColor: 'white',
                backgroundColor: alpha(theme.palette.common.white, 0.1),
              },
            }}
          >
            Go to Dashboard
          </StyledButton>
        </Box>
      </Container>
    </Box>
  );
};

export default Package;
