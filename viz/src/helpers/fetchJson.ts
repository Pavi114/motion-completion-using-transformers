export default async function (url: string): Promise<Object> {
    const response = await fetch(url);

    const json = await response.json();

    return json;
}