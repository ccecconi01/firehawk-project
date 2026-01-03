const express = require('express');
const router = express.Router();
const pool = require('./db');

router.post('/login', async (req, res) => {
  let connection;

  try {
    const { departmentId, password, userType } = req.body;
    console.log('LOGIN BODY:', { departmentId, password, userType });

    // Validate input
    if (!departmentId || !userType) {
      return res.status(400).json({
        success: false,
        message: 'Department ID and user type are required'
      });
    }

    const normalizedType =
      (userType || '').toLowerCase() === 'viewer' ? 'Viewer' : 'operator';

    // For Operator, password is required
    if (normalizedType === 'operator' && !password) {
      return res.status(400).json({
        success: false,
        message: 'Password is required for Operator login'
      });
    }

    // Get connection from pool
    connection = await pool.getConnection();

    // Query database
    const [rows] = await connection.query(
      `
      SELECT *
      FROM Unidade
      WHERE ID_Unidade_User = ?
        AND LOWER(TRIM(Tipo_Utilizador)) = ?
      `,
      [departmentId, normalizedType] // 'viewer' or 'operator'
    );

    console.log('DB ROWS:', rows);

    // Check if user exists
    if (rows.length === 0) {
      return res.status(401).json({
        success: false,
        message: 'Invalid Department ID or User Type'
      });
    }

    const user = rows[0];

    // For Operator, check password
    if (normalizedType === 'operator') {
      if (user.Password_Utilizador !== password) {
        return res.status(401).json({
          success: false,
          message: 'Invalid password'
        });
      }
    }
    // For Viewer, no password check needed

    // Login successful
    return res.json({
      success: true,
      message: 'Login successful',
      user: {
        id: user.ID_Unidade_User,
        departmentId: user.ID_Unidade_User,
        userType: normalizedType, // 'viewer' or 'operator'
        manRegistados: user.Man_Registados,
        terrestresRegistados: user.Terrestres_Registados
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  } finally {
    if (connection) {
      connection.release();
    }
  }
});


router.post('/update-password', async (req, res) => {
  let connection;

  try {
    const { departmentId, userType, currentPassword, newPassword } = req.body;
    console.log('UPDATE PASSWORD BODY:', { departmentId, userType });

    // Validate input
    if (!departmentId || !userType || !currentPassword || !newPassword) {
      return res.status(400).json({
        success: false,
        message: 'All fields are required'
      });
    }

    // Validate new password length
    if (newPassword.length < 4) {
      return res.status(400).json({
        success: false,
        message: 'New password must be at least 4 characters'
      });
    }

    const normalizedType =
      (userType || '').toLowerCase() === 'viewer' ? 'Viewer' : 'operator';

    // Only operators can change password
    if (normalizedType !== 'operator') {
      return res.status(403).json({
        success: false,
        message: 'Only operators can change password'
      });
    }

    // Get connection from pool
    connection = await pool.getConnection();

    // Query database to find user
    const [rows] = await connection.query(
      `
      SELECT *
      FROM Unidade
      WHERE ID_Unidade_User = ?
        AND LOWER(TRIM(Tipo_Utilizador)) = ?
      `,
      [departmentId, normalizedType]
    );

    // Check if user exists
    if (rows.length === 0) {
      return res.status(401).json({
        success: false,
        message: 'Invalid Department ID or User Type'
      });
    }

    const user = rows[0];

    // Verify current password
    if (user.Password_Utilizador !== currentPassword) {
      return res.status(401).json({
        success: false,
        message: 'Current password is incorrect'
      });
    }

    // Update password
    await connection.query(
      `
      UPDATE Unidade
      SET Password_Utilizador = ?
      WHERE ID_Unidade_User = ?
        AND LOWER(TRIM(Tipo_Utilizador)) = ?
      `,
      [newPassword, departmentId, normalizedType]
    );

    return res.json({
      success: true,
      message: 'Password updated successfully'
    });
  } catch (error) {
    console.error('Update password error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    });
  } finally {
    if (connection) {
      connection.release();
    }
  }
});

module.exports = router;