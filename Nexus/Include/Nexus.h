#ifndef NexusHeader
#include <vector>
#define NexusHeader

namespace Nexus
{
	struct Profile;

	struct Synaspe
	{
		unsigned int connection;
		double weight = 0.0, bias = 0.0;
		Synaspe() {}
		Synaspe(unsigned int connection, double weight, double bias) : connection(connection), weight(weight), bias(bias) {}
		static Synaspe Random(Profile& profile);
	};

	struct Bridge
	{
		unsigned int input, output, index;
		Bridge(unsigned int input, unsigned int output, unsigned int index) : input(input), output(output), index(index) {}
	};

	struct Neuron
	{
		double bias = 0.0;
		std::vector<Synaspe> connections;
		bool Connected(unsigned int neuron, unsigned int* index = nullptr);
		double Activate(double memory, double weight = 1.0f);
		Neuron() {}
		Neuron(double bias, std::vector<Synaspe> connections) : bias(bias), connections(connections) {}
		static Neuron Random(Profile& profile);
	};

	struct Profile
	{
		std::vector<Neuron> neurons;
		bool Valid(unsigned int neuron); 
		Profile Random(unsigned int maxNeurons);
		std::vector<Bridge> ConnectionsTo(unsigned int neuron);
	};

	class Network
	{
	private:
		Profile m_profile;
		std::vector<double> m_memory;
	public:
		Network(){}
		Network(const Profile& profile);
		void Wipe(unsigned int neuron);
		void Stream(unsigned int neuron);
		double Read(unsigned int neuron);
		void Feed(unsigned int neuron, double input);
		Profile& Breakdown();
	};

	namespace Compute 
	{
		double Clamp(double x, double a = 0, double b = 1);
		double Range(double percentage, double a, double b);

		int Random(int min, int max);
		double Random(double min = -1, double max = 1);

		double Sigmoid(double x);
		double SigmoidDerivative(double x);

		bool Mutate(double mutation);
		double Mutate(double a, double mutation);
		int Mutate(int a, unsigned int offset, double mutation);
		Synaspe Mutate(Profile* profile, Synaspe synaspe, double mutation);
		Neuron Mutate(Profile* profile, Neuron neuron, double mutation);
		Profile Mutate(Profile& profile, double mutation);

		double Cross(double a, double b);
		Synaspe Cross(Profile* profile, Synaspe a, Synaspe b);
		Neuron Cross(Profile* profile, Neuron a, Neuron b);
		Profile Cross(Profile& a, Profile& b);

		void Configure(Profile* profile, unsigned int index, const double& delta);
		std::vector<Profile> CrossPopulation(std::vector<Profile> population, unsigned int outputSize);
	}
}
#endif