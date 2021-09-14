FROM node:alpine as build

# set the working direction

# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY package.json ./

COPY package-lock.json ./

RUN npm install

EXPOSE 3000


# add app
COPY . ./

# start app
CMD ["npm", "start"]

# production environment
FROM nginx:stable-alpine
COPY --from=build /app2/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]