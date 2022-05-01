export default async function generateMotionSequence(gpos: number[][][]) {
    const response = await fetch('http://127.0.0.1:5000/generate', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            gpos
        }),
        mode: 'cors'
    })

    const json = await response.json()

    return json
}