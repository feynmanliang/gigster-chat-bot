import d3 from 'd3';

class Features {
    constructor(element, height, width, data) {
        this.data = data;
        this.element = element;
        this.updateParts(height, width, data.length);
        if (this.element) {
            this.createGraph();
        }
    }

    updateElt(elt) {
        this.element = elt;
        this.createGraph();
    }

    createGraph() {
        this.svg = d3.select(this.element).append("svg")
            .attr("width", this.width)
            .attr("height", this.height);
        this.draw(this.data);
    }

    updateParts(height, width, dataLength) {
        this.height = height;
        this.width = width;

        this.x = d3.scale.linear()
                    .domain([0, 2])
                    .range([0, this.width]);

        this.y = d3.scale.linear()
            .domain([1, 0])
            .range([this.height, 0]);

        this.xAxis = d3.svg.axis()
            .scale(this.x)
            .orient("bottom");

        this.yAxis = d3.svg.axis()
            .scale(this.y)
            .orient("left")
            .ticks(10, "%");
    }

    draw(data) {
        let f = this;

        this.svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + f.height + ")")
            .call(this.xAxis);

        this.svg.append("g")
            .attr("class", "y axis")
            .call(f.yAxis)
          .append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 6)
            .attr("dy", ".71em")
            .style("text-anchor", "end")
            .text("Frequency");

        this.svg.selectAll(".bar")
            .data(this.data)
            .enter().append("rect")
              .attr("fill", "greenyellow")
              .attr("class", "bar")
              .attr("x", function(d) { return f.x(d.i); })
              .attr("width", this.width/f.data.length)
              .attr("y", function(d) { return f.y(d.data); })
              .attr("height", function(d) { return f.height - f.y(d.data); });

        this.svg.exit().remove();
    }
}

export default Features;
