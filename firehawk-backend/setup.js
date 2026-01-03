require('dotenv').config();
const mysql = require('mysql2/promise');

async function setupDatabase() {
  const dbConfig = {
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
  };

  console.log('...Connecting to MySQL...');

  try {
    // 1. Connect to MySQL (no specific database) to create the DB
    const connection = await mysql.createConnection(dbConfig);
    
    // 2. Create the database
    await connection.query(`CREATE DATABASE IF NOT EXISTS ${process.env.DB_NAME}`);
    console.log(` OK Database '${process.env.DB_NAME}' verified/created.`);
    await connection.end();

    // 3. Connect now to the correct database
    const pool = mysql.createPool({
      ...dbConfig,
      database: process.env.DB_NAME
    });

    // 4. Create table 'Unidade' (per PDF page 7)
    // Note: The PDF uses 'INT' for IDs and passwords for Operators.
    const createTableQuery = `
      CREATE TABLE IF NOT EXISTS Unidade (
        ID_Unidade_User INT,
        Tipo_Utilizador VARCHAR(56) NOT NULL,
        Password_Utilizador VARCHAR(255),
        Man_Registados INT DEFAULT 0,
        Terrestres_Registados INT DEFAULT 0,
        PRIMARY KEY (ID_Unidade_User, Tipo_Utilizador)
      )
    `;
    await pool.query(createTableQuery);
    console.log("OK Table 'Unidade' verified/created.");

    // 5. Clear old data (avoid duplicates on reruns)
    await pool.query('DELETE FROM Unidade');

    // 6. Insert initial users (per PDF page 8)
    const insertQuery = `
      INSERT INTO Unidade (ID_Unidade_User, Tipo_Utilizador, Password_Utilizador, Man_Registados, Terrestres_Registados)
      VALUES ?
    `;

    const users = [
      // Department 1
      [1, 'Operator', 'password123', 15, 5],
      [1, 'Viewer', null, 15, 5],
      // Department 2
      [2, 'Operator', 'password789', 20, 8],
      [2, 'Viewer', null, 20, 8],
      // Department 3
      [3, 'Operator', 'password456', 12, 3],
      [3, 'Viewer', null, 12, 3]
    ];

    await pool.query(insertQuery, [users]);
    console.log("OK USERS - Default users inserted successfully");

    console.log("SETUP COMPLETE! Database is ready.");
    process.exit(0);

  } catch (error) {
    console.error("Error configuring database:", error);
    process.exit(1);
  }
}

setupDatabase();