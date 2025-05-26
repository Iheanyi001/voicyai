import React, { useState } from "react";
import { useNavigate, Link as RouterLink } from "react-router-dom";
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Link from '@mui/material/Link';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import { alpha, styled } from '@mui/material/styles';

// Styled components
const StyledPaper = styled(Paper)(({ theme }) => ({
  background: alpha(theme.palette.background.paper, 0.9),
  backdropFilter: 'blur(10px)',
  borderRadius: '24px',
  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
  maxWidth: '450px',
  width: '100%',
  margin: '0 auto',
  padding: theme.spacing(4),
}));

const StyledTextField = styled(TextField)(({ theme }) => ({
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
  '& .MuiInputLabel-root': {
    fontSize: '1rem',
  },
  '& .MuiOutlinedInput-input': {
    fontSize: '1rem',
    padding: '16px',
  },
}));

const StyledButton = styled(Button)(({ theme }) => ({
  borderRadius: '12px',
  padding: '12px 0',
  textTransform: 'none',
  fontSize: '1.1rem',
  fontWeight: 600,
  boxShadow: '0 4px 12px 0 rgba(0,0,0,0.1)',
  '&:hover': {
    boxShadow: '0 6px 16px 0 rgba(0,0,0,0.2)',
    transform: 'translateY(-1px)',
  },
  transition: 'all 0.2s ease-in-out',
}));

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const validateForm = () => {
    if (!email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      setError("Please enter a valid email address");
      return false;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters long");
      return false;
    }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);
    
    try {
      if (!validateForm()) {
        setIsLoading(false);
        return;
      }

      const res = await fetch("/api/login", {
        method: "POST",
        body: new URLSearchParams({ email, password }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || "Login failed");
      }

      // Store token and user data
      localStorage.setItem("token", data.access_token);
      localStorage.setItem("user_type", data.user_type);
      localStorage.setItem("name", data.name);
      localStorage.setItem("first_login", "true");

      // Verify token before proceeding
      const verifyRes = await fetch("/api/verify-token", {
        headers: {
          'Authorization': `Bearer ${data.access_token}`
        }
      });

      if (!verifyRes.ok) {
        throw new Error("Token verification failed");
      }

      navigate("/package");
    } catch (err) {
      setError(err.message);
      // Clear any partial data on error
      localStorage.removeItem("token");
      localStorage.removeItem("user_type");
      localStorage.removeItem("name");
      localStorage.removeItem("first_login");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box sx={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px',
    }}>
      <StyledPaper elevation={6}>
        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          width: '100%',
        }}>
          <Avatar sx={{ 
            m: 1, 
            bgcolor: 'primary.main',
            width: 64,
            height: 64,
            boxShadow: '0 4px 12px 0 rgba(0,0,0,0.1)',
          }}>
            <LockOutlinedIcon fontSize="large" />
          </Avatar>
          <Typography component="h1" variant="h4" sx={{ 
            mt: 2,
            mb: 4,
            fontWeight: 700,
            background: 'linear-gradient(45deg, #667eea, #764ba2)',
            backgroundClip: 'text',
            textFillColor: 'transparent',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textAlign: 'center',
          }}>
            Welcome Back
          </Typography>
          <Box component="form" onSubmit={handleSubmit} sx={{ width: '100%' }}>
            <StyledTextField
              margin="normal"
              required
              fullWidth
              id="email"
              label="Email Address"
              name="email"
              autoComplete="email"
              autoFocus
              value={email}
              onChange={e => setEmail(e.target.value)}
              error={error && error.includes("email")}
              helperText={error && error.includes("email") ? error : ""}
              disabled={isLoading}
            />
            <StyledTextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              autoComplete="current-password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              error={error && error.includes("Password")}
              helperText={error && error.includes("Password") ? error : ""}
              disabled={isLoading}
            />
            {error && !error.includes("email") && !error.includes("Password") && 
              <Typography color="error" sx={{ mt: 2, textAlign: 'center' }}>{error}</Typography>
            }
            <StyledButton
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 4, mb: 3 }}
              disabled={isLoading}
            >
              {isLoading ? "Signing In..." : "Sign In"}
            </StyledButton>
            <Box sx={{ textAlign: 'center' }}>
              <Link 
                component={RouterLink} 
                to="/register" 
                variant="body1"
                sx={{
                  color: 'primary.main',
                  textDecoration: 'none',
                  fontSize: '1rem',
                  fontWeight: 500,
                  '&:hover': {
                    textDecoration: 'underline',
                  },
                }}
              >
                {"Don't have an account? Sign Up"}
              </Link>
            </Box>
          </Box>
        </Box>
      </StyledPaper>
    </Box>
  );
};

export default Login;
