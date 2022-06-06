mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"imrannazir.485.9@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml