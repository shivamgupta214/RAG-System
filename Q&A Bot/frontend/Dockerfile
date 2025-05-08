# frontend/Dockerfile
FROM node:18

# Set working directory
WORKDIR /app

# Copy files and install deps
COPY . .
RUN npm install
RUN npm run build

# Serve the static files
RUN npm install -g serve
EXPOSE 3000
CMD ["serve", "-s", "build", "-l", "3000"]