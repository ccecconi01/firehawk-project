# 1. Imagem Híbrida (Python + Node)
FROM nikolaik/python-nodejs:python3.10-nodejs18

# 2. Configura pasta de trabalho
WORKDIR /app

# 3. Copia tudo para dentro
COPY . .

# 4. Instala dependências do Python (Pipeline + Server)
WORKDIR /app/AI_FirehawkLab
RUN pip install -r requirements.txt

# 5. Constrói o Frontend (React)
WORKDIR /app/frontend
RUN npm install
# Se usar Vite é 'npm run build', se Create-React-App também.
# Isso cria a pasta 'dist' ou 'build'. Vamos assumir 'build' (padrão CRA) ou 'dist' (Vite).
# VERIFIQUE SE O SEU PACKAGE.JSON DIZ "build": "vite build" ou similar.
RUN npm run build

# 6. Volta para o Python e expõe a porta
WORKDIR /app/AI_FirehawkLab
EXPOSE 5001

# 7. Roda o servidor Python (que vai servir o site também)
CMD ["python", "server.py"]