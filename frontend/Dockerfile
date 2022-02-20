# pull base image
FROM node:16

# set working directory
WORKDIR /app

# install app dependencies

COPY package.json ./

RUN npm install

# add app
COPY . .

#start app

CMD ["npm", "start"]