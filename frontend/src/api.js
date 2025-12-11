import axios from 'axios';

const API_URL = 'http://localhost:5001/api';

export const fetchInitialData = async () => {
    const response = await axios.get(`${API_URL}/initial-data`);
    return response.data;
};

export const updateModel = async (observations) => {
    const response = await axios.post(`${API_URL}/update`, { observations });
    return response.data;
};

export const resetModel = async () => {
    const response = await axios.post(`${API_URL}/reset`);
    return response.data;
};

export const ingestRealResult = async (payload) => {
    const response = await axios.post(`${API_URL}/ingest-real`, payload);
    return response.data;
};
