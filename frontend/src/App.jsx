import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from "react-router-dom";
import Register from "./Register";
import Login from "./Login";
import Package from "./Package";
import Dashboard from "./Dashboard";
import Payment from "./Payment";
import WelcomePage from "./components/WelcomePage";
import "./App.css";

function App() {
  const token = localStorage.getItem("token");
  const firstLogin = localStorage.getItem("first_login");

  // Custom logic for root redirect
  function RootRedirect() {
    if (!token) return <WelcomePage />;
    if (firstLogin === "true") return <Navigate to="/package" />;
    return <Navigate to="/dashboard" />;
  }

  return (
    <Router>
      <Routes>
        <Route path="/" element={<WelcomePage />} />
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />
        <Route path="/package" element={token ? <Package /> : <Navigate to="/login" />} />
        <Route path="/dashboard" element={token ? <Dashboard /> : <Navigate to="/login" />} />
        <Route path="/payment" element={token ? <Payment /> : <Navigate to="/login" />} />
        <Route path="*" element={<RootRedirect />} />
      </Routes>
    </Router>
  );
}

export default App;
