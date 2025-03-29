import Layout from '@/components/Layout'
import PredictionChart from '@/components/PredictionChart'

export default function Home() {
  return (
    <Layout>
      <div className="max-w-7xl mx-auto py-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">
          ETH Price & Gas Predictor
        </h1>
        <PredictionChart />
      </div>
    </Layout>
  )
}
