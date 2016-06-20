import React from 'react';

export default class Header extends React.Component {
  render() {
    return (
      <div className="channel-header">
        <h1 className="ui dividing header channel-header_name">
          <span className="channel-header_prefix">@</span>
          {this.props.channel}
        </h1>
      </div>
    );
  }
}

Header.propTypes = {
  channel: React.PropTypes.string.isRequired
}
